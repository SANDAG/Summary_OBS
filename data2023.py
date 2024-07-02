"""Module for preparing data related to 2023 Onboard survey.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import toml


def _recode_series(
    series: pd.Series,
    codebook: pd.DataFrame,
    variable: str,
) -> pd.Series:
    return (
        series
        .astype(
            pd.CategoricalDtype(
                categories=codebook.loc[variable, :].index,
                ordered=True,
            )
        )
        .cat.rename_categories(
            pd.Series(codebook.loc[variable, 'label'])
        )
    )


def extract_routes(config: dict) -> pd.DataFrame:
    paths = config['obs_2023']['paths']
    trrt_csv_path = Path(paths['trrt_dir'])/paths['trrt_csv']
    return (
        pd.concat(
            [
                (
                    pd.read_csv(trrt_csv_path)
                    .rename(
                        columns={
                            'Route_Name': 'route',
                            'Mode': 'mode',
                        }
                    )
                    .assign(
                        route=lambda df: df.route
                        .astype('string').str[:-3]
                        .astype('int')
                    )
                    .assign(
                        mode=lambda df: df['mode'].map(
                            # per https://github.com/SANDAG/ABM/wiki/
                            # input-files
                            {
                                4: 'Commuter Rail',  # coaster
                                5: 'LRT',  # sprinter/trolley
                                6: 'Rapid',  # rapid bus
                                7: 'Rapid',  # rapid bus
                                8: 'Express',  # prem express
                                9: 'Express',  # regular express
                                10: 'Local',  # local bus
                            }
                        )
                    )
                    [['route', 'mode']]
                    .drop_duplicates()
                    .set_index('route')
                    .sort_index()
                ),
                pd.DataFrame(
                    [
                        {'route': 651, 'mode': 'Local'},
                        {'route': 652, 'mode': 'Local'},
                        {'route': 280, 'mode': 'Express'},
                        {'route': 888, 'mode': 'Local'},
                        {'route': 891, 'mode': 'Local'},
                        {'route': 892, 'mode': 'Local'},
                        {'route': 950, 'mode': 'Express'},
                    ]
                ).set_index('route')
            ]
        )
        .astype(
            {
                'mode': pd.CategoricalDtype(
                    categories=[
                        'Local', 'Rapid', 'Express', 'LRT', 'Commuter Rail'
                    ],
                    ordered=True,
                )
            }
        )
    )


def extract_codebook(config: dict) -> pd.DataFrame:
    paths = config['obs_2023']['paths']
    codebook_xlsx = Path(paths['obs_dir'])/paths['od_xlsx']
    codebook_sheet = 'data dictionary'
    return (
        pd.read_excel(codebook_xlsx, sheet_name=codebook_sheet)
        .rename(
            columns={
                'FIELD NAME': 'variable',
                'DESCRIPTION': 'description',
                'CODE VALUES': 'value_label',
            }
        )
        .assign(
            value_label=lambda df: df.value_label.apply(
                lambda x: x if 'Actual Value' not in x else None
            )
        )
        .dropna(subset='value_label')
        .assign(
            value=lambda df: df.value_label
            .apply(lambda x: x.split('=')[0].strip())
        )
        .assign(
            label=lambda df: df.value_label
            .apply(lambda x: x.split('=')[1].strip())
        )
        .drop(columns=['value_label', 'description'])
        .set_index(['variable', 'value'])
    )


def extract_results(config: dict) -> pd.DataFrame:
    paths = config['obs_2023']['paths']
    obs_od_parquet = Path(paths['obs_dir'])/'obs2023_od_results.parquet'
    obs_od_xlsx = Path(paths['obs_dir'])/paths['od_xlsx']
    results_sheet = 'OD_RESULTS'

    routes = extract_routes(config)
    codebook = extract_codebook(config)

    if not obs_od_parquet.exists():
        (
            pd.read_excel(obs_od_xlsx, sheet_name=results_sheet)
            .set_index('ID')
            .astype('string')
            .to_parquet(obs_od_parquet)  # type: ignore
        )

    results = pd.read_parquet(obs_od_parquet)

    return (
        pd.concat(
            {
                'route': _extract_route_data(results, routes),
                'age': _extract_age_data(results),
                'income': _extract_income_data(results, codebook),
                'employment': _extract_employment_data(results, codebook),
                'student': _extract_student_data(results, codebook),
                'weight': _extract_weight_data(results),
            },
            axis=1,
        )
    )


def _extract_route_data(results: pd.DataFrame, routes) -> pd.DataFrame:
    route = results['ROUTE_DIRECTION[Code]']
    return (
        pd.DataFrame(
            data={'route': route},
        )
        .assign(route=lambda df: df.route.str.replace('Blue', '510'))
        .assign(route=lambda df: df.route.str.replace('Orange', '520'))
        .assign(route=lambda df: df.route.str.replace('Green', '530'))
        .assign(route=lambda df: df.route.apply(
            lambda x: int(x.split('_')[2]))
        )
        .reset_index()
        .merge(routes, how='left', on='route')
        .set_index('ID')
    )


def _extract_age_data(results: pd.DataFrame) -> pd.DataFrame:
    date_completed = pd.to_datetime(results.DATE_COMPLETED).dt.year
    year_born = results.YEAR_BORN.astype('float').astype('Int64')
    return (
        pd.DataFrame(
            data={'age': (date_completed - year_born)},
        )
        .assign(
            hhts_age=lambda results: pd.cut(
                results.age,
                bins=[0, 4, 15, 17, 18, 34, 44, 54, 64, 74, 84, np.inf],
                labels=[
                    'Under 5',
                    '5-15',
                    '16-17',
                    '17-18',
                    '25-34',
                    '35-44',
                    '45-54',
                    '55-64',
                    '65-74',
                    '75-84',
                    '85 or Older',
                ],
            )
        )
        .assign(
            yas=lambda results: pd.cut(
                results.age,
                bins=[0, 18, 64, np.inf],
                labels=['Youth', 'Adult', 'Senior'],
            )
        )
    )


def _extract_income_data(
    results: pd.DataFrame,
    codebook: pd.DataFrame,
) -> pd.DataFrame:
    hh_income = _recode_series(
        series=results['INCOME[Code]'],
        codebook=codebook,
        variable='INCOME',
    )
    return (
        pd.DataFrame(
            data={'hh_income': hh_income},
        )
    )


def _extract_employment_data(
    results: pd.DataFrame,
    codebook: pd.DataFrame,
) -> pd.DataFrame:
    employment_status = _recode_series(
        series=results['EMPLOYMENT_STATUS[Code]'],
        codebook=codebook,
        variable='EMPLOYMENT_STATUS',
    )
    hh_employed = _recode_series(
        series=results['EMPLOYED_IN_HH[Code]'],
        codebook=codebook,
        variable='EMPLOYED_IN_HH',
    )
    return (
        pd.DataFrame(
            data={
                'employment_status': employment_status,
                'hh_employed': hh_employed,
            },
        )
    )


def _extract_student_data(
    results: pd.DataFrame,
    codebook: pd.DataFrame,
) -> pd.DataFrame:
    student_status = _recode_series(
        series=results['STUDENT_STATUS[Code]'],
        codebook=codebook,
        variable='STUDENT_STATUS',
    )
    return (
        pd.DataFrame(
            data={
                'student_status': student_status,
            },
        )
    )


def _extract_weight_data(results: pd.DataFrame) -> pd.DataFrame:
    unlinked_weight = results.UNLINKED_WGHT_FCTR.astype('float')
    linked_weight = results.LINKED_WGHT_FCTR.astype('float')
    return (
        pd.DataFrame(
            data={
                'unlinked_weight': unlinked_weight,
                'linked_weight': linked_weight,
            },
        )
    )


if __name__ == '__main__':
    config = toml.load('config.toml')
    obs2023 = extract_results(config)
    obs2023.to_parquet('./data/obs2023.parquet')
