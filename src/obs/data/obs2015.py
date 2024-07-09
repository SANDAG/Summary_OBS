from dataclasses import dataclass

from functools import cached_property
from pathlib import Path

import pandas as pd
from loguru import logger


@dataclass
class OBS2015Config:
    routes_path: Path
    results_path: Path
    results_sheet: str
    data_dictionary_path: Path
    data_dictionary_sheet: str
    save_dir: Path


@dataclass
class OBS2015:
    config: OBS2015Config

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
            self.results[["ROUTE_SURVEYED_CODE"]]
            .rename(columns={"ROUTE_SURVEYED_CODE": "route"})
            .assign(route=lambda df: df.route.str[1:4].astype("int"))
            .reset_index()
            .merge(self.routes, how="left", on="route")
            # TODO: Assume Local for now, need to look up historic route #s
            .assign(mode=lambda df: df["mode"].fillna("Local"))
            .set_index("ID")
        )

    @property
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
                        (
                            pd.to_datetime(
                                self.results["DATE"],
                                errors="coerce",
                            ).dt.year
                        )
                        - self.results["YEAR_OF_BIRTH"].astype("Int32")
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
                "unlinked_weight": (
                    self.results["UNLINKED_WEIGHT_FACTOR"].astype("float")
                ),
                "linked_weight": (
                    self.results["FACTOR_TO_EXPAND_TO_LINKED_TRIPS"].astype("float")
                ),
            }
        )

    def save(self) -> None:
        self.routes.to_parquet(self.config.save_dir / "routes.parquet")
        self.results.to_parquet(self.config.save_dir / "results.parquet")
        self.transit_mode.to_parquet(self.config.save_dir / "transit_mode.parquet")
        self.age.to_parquet(self.config.save_dir / "age.parquet")
        self.weights.to_parquet(self.config.save_dir / "weights.parquet")
