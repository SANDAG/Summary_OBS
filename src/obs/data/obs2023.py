from dataclasses import dataclass

from functools import cached_property
from pathlib import Path

import pandas as pd
from loguru import logger


@dataclass
class OBS2023Config:
    routes_path: Path
    results_path: Path
    results_sheet: str
    data_dictionary_path: Path
    data_dictionary_sheet: str
    save_dir: Path


@dataclass
class OBS2023:
    config: OBS2023Config

    @cached_property
    def routes(self) -> pd.DataFrame:
        logger.info("Loading and caching `routes` data.")
        return pd.concat(
            [
                (
                    pd.read_csv(self.config.routes_path)
                    .rename(
                        columns={
                            "Route_Name": "route",
                            "Mode": "mode",
                        }
                    )
                    .assign(
                        route=lambda df: df.route.astype("string")
                        .str[:-3]
                        .astype("int")
                    )
                    .assign(
                        mode=lambda df: df["mode"].map(
                            # per https://github.com/SANDAG/ABM/wiki/
                            # input-files
                            {
                                4: "Commuter Rail",  # coaster
                                5: "LRT",  # sprinter/trolley
                                6: "Rapid",  # rapid bus
                                7: "Rapid",  # rapid bus
                                8: "Express",  # prem express
                                9: "Express",  # regular express
                                10: "Local",  # local bus
                            }
                        )
                    )[["route", "mode"]]
                    .drop_duplicates()
                    .set_index("route")
                    .sort_index()
                ),
                pd.DataFrame(
                    [
                        {"route": 651, "mode": "Local"},
                        {"route": 652, "mode": "Local"},
                        {"route": 280, "mode": "Express"},
                        {"route": 888, "mode": "Local"},
                        {"route": 891, "mode": "Local"},
                        {"route": 892, "mode": "Local"},
                        {"route": 950, "mode": "Express"},
                    ]
                ).set_index(["route"]),
            ]
        ).astype(
            {
                "mode": pd.CategoricalDtype(
                    categories=["Local", "Rapid", "Express", "LRT", "Commuter Rail"],
                    ordered=True,
                )
            }
        )

    @cached_property
    def results(self) -> pd.DataFrame:
        logger.info("Loading and caching `result` data.")
        # todo, load parquet if availible
        return pd.read_excel(
            self.config.results_path,
            sheet_name=self.config.results_sheet,
            dtype="string[pyarrow]",
        ).set_index("ID")

    @property
    def transit_mode(self) -> pd.DataFrame:
        return (
            self.results[["ROUTE_DIRECTION[Code]"]]
            .rename(columns={"ROUTE_DIRECTION[Code]": "route"})
            .assign(route=lambda df: df.route.str.replace("Blue", "510"))
            .assign(route=lambda df: df.route.str.replace("Orange", "520"))
            .assign(route=lambda df: df.route.str.replace("Green", "530"))
            .assign(route=lambda df: df.route.apply(lambda x: int(x.split("_")[2])))
            .reset_index()
            .merge(self.routes, how="left", on="route")
            .rename(columns={"mode": "transit_mode"})
            .set_index("ID")
        )

    @property
    def access_egress_mode(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "access_mode": (
                    self.results["ORIGIN_TRANSPORT"].astype(
                        pd.CategoricalDtype(
                            categories=[
                                "Walk",
                                "Wheelchair",
                                "Bike (personal)",
                                "E-Bike (personal)",
                                "E-Bike (shared)",
                                "Skateboard",
                                "E-scooter (personal)",
                                "E-scooter (shared)",
                                "Uber, Lyft, etc. (private)",
                                "Uber, Lyft, etc. (pool or shared)",
                                "Taxi",
                                "Was dropped off by someone",
                                "Drove alone and parked",
                                "Drove or rode with others and parked",
                                "Electric vehicle shuttle",
                                "Other shuttle",
                                "Other",
                            ],
                            ordered=True,
                        )
                    )
                ),
                "egress_mode": (
                    self.results["DESTIN_TRANSPORT"].astype(
                        pd.CategoricalDtype(
                            categories=[
                                "Walk",
                                "Wheelchair",
                                "Bike (personal)",
                                "E-Bike (personal)",
                                "E-Bike (shared)",
                                "Skateboard",
                                "E-scooter (personal)",
                                "E-scooter (shared)",
                                "Uber, Lyft, etc. (private)",
                                "Uber, Lyft, etc. (pool or shared)",
                                "Taxi",
                                "Be picked up by someone",
                                "Get in a parked vehicle & drive alone",
                                "Get in a parked vehicle & drive/ride w/others",
                                "Electric vehicle shuttle",
                                "Other shuttle",
                                "Other",
                                "Refused/No Answer",
                            ],
                            ordered=True,
                        )
                    )
                ),
            }
        ).assign(
            access_mode_abm=lambda df: df["access_mode"]
            .astype("string")
            .map(
                {
                    "Walk": "Walk to transit",
                    "Wheelchair": "Walk to transit",
                    "Bike (personal)": "Bike to transit",
                    "E-Bike (personal)": "Micromobility to transit",
                    "E-Bike (shared)": "Micromobility to transit",
                    "Skateboard": "Walk to transit",
                    "E-scooter (personal)": "Micromobility to transit",
                    "E-scooter (shared)": "Micromobility to transit",
                    "Uber, Lyft, etc. (private)": "TNC to transit",
                    "Uber, Lyft, etc. (pool or shared)": "TNC to transit",
                    "Taxi": "TNC to transit",
                    "Was dropped off by someone": "KNR to transit",
                    "Drove alone and parked": "PNR to transit",
                    "Drove or rode with others and parked": "PNR to transit",
                    "Electric vehicle shuttle": "TNC to transit",
                    "Other shuttle": "TNC to transit",
                    "Other": None,
                }
            )
            .astype(
                pd.CategoricalDtype(
                    categories=[
                        "Walk to transit",
                        "Bike to transit",
                        "Micromobility to transit",
                        "PNR to transit",
                        "KNR to transit",
                        "TNC to transit",
                    ],
                    ordered=True,
                ),
            )
        )

    @cached_property
    def age(self) -> pd.DataFrame:
        def map_age_category(age: int | None) -> str | None:
            if age and not pd.isna(age):
                if age < 5:
                    return "Under 5"  # noqa: E701
                elif age < 16:
                    return "5-15"  # noqa: E701
                elif age < 18:
                    return "16-17"  # noqa: E701
                elif age < 25:
                    return "18-24"  # noqa: E701
                elif age < 35:
                    return "25-34"  # noqa: E701
                elif age < 45:
                    return "35-44"  # noqa: E701
                elif age < 55:
                    return "45-54"  # noqa: E701
                elif age < 65:
                    return "55-64"  # noqa: E701
                elif age < 75:
                    return "65-74"  # noqa: E701
                elif age < 85:
                    return "75-84"  # noqa: E701
                else:
                    return "85 or Older"  # noqa: E701
            else:
                return None

        def map_yas(age: int | None) -> str | None:
            if age and not pd.isna(age):
                if age < 18:
                    return "Youth"  # noqa: E701
                elif age < 65:
                    return "Adult"  # noqa: E701
                else:
                    return "Senior"  # noqa: E701
            else:
                return None

        return (
            pd.DataFrame(
                {
                    "age": (
                        pd.to_datetime(self.results["DATE_COMPLETED"]).dt.year
                        - self.results["YEAR_BORN"].astype("Int32")
                    )
                }
            )
            .assign(age_category=lambda df: df.age.map(map_age_category))
            .assign(
                age_category=lambda df: df.age_category.astype(
                    pd.CategoricalDtype(
                        [
                            "Under 5",
                            "5-15",
                            "16-17",
                            "18-24",
                            "25-34",
                            "35-44",
                            "45-54",
                            "55-64",
                            "65-74",
                            "75-84",
                            "85 or Older",
                        ],
                        ordered=True,
                    )
                )
            )
            .assign(age_yas=lambda df: df.age.map(map_yas))
            .assign(
                age_yas=lambda df: df.age_yas.astype(
                    pd.CategoricalDtype(
                        ["Youth", "Adult", "Senior"],
                        ordered=True,
                    )
                )
            )
        )

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unlinked_weight": (self.results["UNLINKED_WGHT_FCTR"].astype("float")),
                "linked_weight": (self.results["LINKED_WGHT_FCTR"].astype("float")),
            }
        )

    def save(self) -> None:
        self.routes.to_parquet(self.config.save_dir / "routes.parquet")
        self.results.to_parquet(self.config.save_dir / "results.parquet")
        self.transit_mode.to_parquet(self.config.save_dir / "transit_mode.parquet")
        self.age.to_parquet(self.config.save_dir / "age.parquet")
        self.access_egress_mode.to_parquet(
            self.config.save_dir / "access_egress_mode.parquet"
        )
        self.weights.to_parquet(self.config.save_dir / "weights.parquet")
