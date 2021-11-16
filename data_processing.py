import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = (24, 12)


def getData():
    # get currently available data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        parse_dates=["date"],
    )

    # select attributes that we find interesting
    attributes = [
        "iso_code",
        "continent",
        "location",
        "date",
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths",
        "reproduction_rate",
        "icu_patients",
        "hosp_patients",
        "weekly_icu_admissions",
        "weekly_hosp_admissions",
        "new_tests",
        "total_tests",
        "positive_rate",
        "total_vaccinations",
        "people_vaccinated",
        "people_fully_vaccinated",
        "new_vaccinations",
        "stringency_index",
        "population",
        "population_density",
        "median_age",
        "aged_65_older",
        "aged_70_older",
        "gdp_per_capita",
        "extreme_poverty",
        "cardiovasc_death_rate",
        "diabetes_prevalence",
        "female_smokers",
        "male_smokers",
        "handwashing_facilities",
        "hospital_beds_per_thousand",
        "life_expectancy",
        "human_development_index",
        "excess_mortality",
    ]

    df = df[attributes]

    # now check which columns contain NaN values

    # it's weird that continent contains NaNs and location doesn't, so let's take a look at that

    # let's store these rows which combine data of a continent in a seperate dataframe, so it doesn't give us any weird mistakes later/confuse us
    # before we do this we should be sure that really only these combined rows have NaNs
    # looking good, so lets create a new dataframe
    continent_df = df[df["continent"].isna()]
    # drop these rows from the original dataframe
    df = df[~df["continent"].isna()]

    # we would expect that new_cases should be almost complete (because it's the most important attribute), so let's take a look at that

    # it seems that there are countries that don't have any values for new_cases or simply not enough values, these are obviously useless to us
    # lets drop all countries that have missing values for more than half of their entries for the column new_cases
    # (if they have more than half of the values, then we can fix the missing values with interpolation later)
    temp = (
        df[["location", "new_cases"]].groupby("location").count()["new_cases"]
        / df[["location", "new_cases"]].groupby("location").size()
    ) >= 0.5
    temp = temp[temp == True].index.tolist()
    # drop all rows that don't fulfill the above defined criteria
    df = df[df["location"].isin(temp)]
    # reset index, so that it is correct again (we dropped rows)
    df.reset_index(inplace=True, drop=True)

    # also while looking through the records, we found that some countries have leading NaNs for new_cases
    # let's remove these (while we're at it, let's also remove trailing NaNs)
    df = df.sort_values(by=["location", "date"]).reset_index(drop=True)
    # get the first and last valid index
    first_valid_index = df.groupby("location").apply(
        lambda x: x["new_cases"].first_valid_index()
    )
    last_valid_index = df.groupby("location").apply(
        lambda x: x["new_cases"].last_valid_index()
    )
    # create list of indices that we want to keep
    valid_indices = [
        np.arange(first, last + 1)
        for first, last in zip(first_valid_index, last_valid_index)
    ]
    # flatten it to be a 1D array instead of 2D
    valid_indices = [elem for sublist in valid_indices for elem in sublist]
    df = df[df.index.isin(valid_indices)]
    # we removed rows, so we need to reset the index
    df.reset_index(drop=True, inplace=True)

    # let's look at what percentage of values is still NaN for each column

    # for some reason total_cases is complete, but new_cases isn't so let's fix that real quick
    temp = (
        df.groupby("location")
        .apply(
            lambda x: np.append(
                x["total_cases"].iloc[1:2].to_numpy(),
                x["total_cases"].diff().to_numpy()[1:],
                axis=0,
            )
        )
        .to_numpy()
    )
    temp = [elem for sublist in temp for elem in sublist]
    df["new_cases"] = temp

    # also it seems that there are some columns where we simply have too many missing values for them to be useful, let's remove these
    cols_to_drop = [
        "icu_patients",
        "hosp_patients",
        "weekly_icu_admissions",
        "weekly_hosp_admissions",
        "excess_mortality",
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # let's try and fill the missing values for the remaining columns
    # vaccinations numbers are very interesting to us so let's take a look at it
    # for each country get the percentage of values that are not NaN for total_vaccinations

    # maybe the high number of missing values comes from leading NaNs? let's check that
    temp = df.copy()
    temp.reset_index(inplace=True)
    first_index = temp.groupby("location").apply(lambda x: x.iloc[0]["index"])
    first_valid_index = temp.groupby("location").apply(
        lambda x: x["total_vaccinations"].first_valid_index()
    )
    (first_valid_index - first_index).to_frame().T

    # so it seems that there are a lot of leading NaNs, however for some countries we don't have any vaccination numbers, lets remove these countries
    temp = first_valid_index - first_index
    # also after removing these countries, we need to reset the index (because we dropped some rows)
    df = df[~df["location"].isin(temp[temp.isna()].index)].reset_index(drop=True)

    # let's look at the first value of total_vaccinations for each country that isn't NaN
    first_valid_index = df.groupby("location").apply(
        lambda x: x["total_vaccinations"].first_valid_index()
    )
    temp = df.iloc[list(first_valid_index)][
        ["total_vaccinations", "location"]
    ].set_index("location")
    # get the countries that have zero as first non-NaN value
    countries_with_zero = temp[temp["total_vaccinations"] == 0].index

    # so unfortunately these aren't always zero, however just interpolating these leading NaNs would temper too much with the given data
    # let's just set the ones to zero where the first non-NaN value is zero
    df.reset_index(inplace=True)
    temp = df[df["location"].isin(countries_with_zero)].groupby("location")
    first_index = temp.apply(lambda x: x.iloc[0]["index"])
    first_valid_index = temp.apply(
        lambda x: x["total_vaccinations"].first_valid_index()
    )
    df.drop(columns=["index"], inplace=True)
    # create list of indices that we want to change
    zero_indices = [
        np.arange(first, last - 1)
        for first, last in zip(first_index, first_valid_index)
    ]
    # flatten it to be a 1D array instead of 2D
    zero_indices = [elem for sublist in zero_indices for elem in sublist]
    # also set the people_vaccinated, people_fully_vaccinated, new_vaccinations to 0 for these rows
    df.loc[
        zero_indices,
        [
            "total_vaccinations",
            "people_vaccinated",
            "people_fully_vaccinated",
            "new_vaccinations",
        ],
    ] = 0

    # let's look at the percentage of missing values for columns again

    # it's looking a lot better now, but the ~50% missing values for new_tests/total_tests are really annoying because these are such interesting columns
    # lets's check for leading NaNs
    temp = df.copy()
    temp.reset_index(inplace=True)
    # get the first index for each country
    first_index = temp.groupby("location").apply(lambda x: x.iloc[0]["index"])
    # get the first valid index for each country for the column total_tests
    first_valid_index = temp.groupby("location").apply(
        lambda x: x["total_tests"].first_valid_index()
    )

    # let's check what the first value looks like that isn't NaN
    df.reset_index(drop=True, inplace=True)
    first_valid_index = df.groupby("location").apply(
        lambda x: x["total_tests"].first_valid_index()
    )
    first_valid_index = first_valid_index.dropna()
    temp = df.iloc[list(first_valid_index)][["total_tests", "location"]].set_index(
        "location"
    )

    # unfortunately it's not zero for basically all of them, so we can't really do much here
    # maybe looking at the percentage of missing values will help us somehow

    # not really sure what to do with these
    # we'll do some interpolation later for these (between first and last valid value for each country) and see to what extent that fixes it
    # let's look at other columns that have a high percentage of missing values

    # this looks similar to total_tests (either countries have a high number of non-NaN values (> 80%) or a low number (< 20%))
    # also not sure what to do with these (we'll also interpolate it later)
    # let's take a look at another column with a high percentage of NaN-values

    # ok this is even more extreme now it's either 100% or 0% now, we can't really do anything here
    # let's take a look at another column with a high percentage of NaN-values

    # we can't really do anything here as well..
    # so let's do some interpolation (we'll only interpolate between the first valid value and the last, because it would probably temper too much with the data)
    # first we'll look at the percentage of missing values for each column again
    cols = df.columns

    # since we're only gonna interpolate between the first and last valid value for each country, we can basically put each column in here and see to what extent it fixes something
    # we'll only interpolate the total_columns and add the missing values later for the new_columns
    cols_to_interpolate = [
        "total_deaths",
        "reproduction_rate",
        "total_tests",
        "positive_rate",
        "total_vaccinations",
        "people_vaccinated",
        "people_fully_vaccinated",
        "stringency_index",
        "population_density",
        "median_age",
        "aged_65_older",
        "aged_70_older",
        "gdp_per_capita",
        "extreme_poverty",
        "cardiovasc_death_rate",
        "diabetes_prevalence",
        "female_smokers",
        "male_smokers",
        "handwashing_facilities",
        "hospital_beds_per_thousand",
        "life_expectancy",
        "human_development_index",
    ]
    df = df.groupby("location").apply(
        lambda x: x[df.columns.difference(cols_to_interpolate)].join(
            x[cols_to_interpolate].interpolate(
                method="linear", axis=0, limit_area="inside"
            )
        )
    )[cols]
    # let's change new_deaths, new_tests and new_vaccinations accordingly now
    df[["new_deaths", "new_tests", "new_vaccinations"]] = (
        df.groupby("location")
        .apply(
            lambda x: x[["total_deaths", "total_tests", "total_vaccinations"]].diff()
        )
        .to_numpy()
    )

    # let's see how that affected the percentage of missing values for each column

    # it might be fine to use the average value of the continent for a country for a specific column if it's completly missing
    # (this only applies to some columns (columns that describe local factors and that we won't expect to change much over the time interval))
    # however we should probably first look at the standard deviation and compare it with the mean and if the std is too big, we can't do it for that column
    cols_to_consider = [
        "location",
        "population_density",
        "median_age",
        "aged_65_older",
        "aged_70_older",
        "extreme_poverty",
        "cardiovasc_death_rate",
        "diabetes_prevalence",
        "female_smokers",
        "male_smokers",
        "handwashing_facilities",
        "hospital_beds_per_thousand",
        "life_expectancy",
        "human_development_index",
    ]
    # get the continent for each country
    continents = df.groupby("location").apply(lambda x: x.iloc[0]["continent"])
    # get the mean value for the considered columns for each country
    temp = df[cols_to_consider].groupby("location").mean()
    # add continent as column
    temp["continent"] = continents
    # now get the relative size of std to mean for each continent and then average that out for all continents
    means = temp.groupby("continent").mean()
    means_all = (
        temp.groupby("continent").std() / temp.groupby("continent").mean()
    ).mean()

    # for some columns the std is quite big compared to the mean value, for these columns it's probably not a good idea to just use the mean (this would temper too much with our data)
    # now we just need to define a threshhold at which we want to use the mean of the continent for missing values
    threshhold = 0.5
    cols_to_use_mean = means_all <= threshhold
    cols_to_use_mean = cols_to_use_mean[cols_to_use_mean].index

    for col in cols_to_use_mean:
        # get all rows that are NaN for this column
        nan_indices = df[col].isna()
        # set it to the mean value of the continent of that country for that column
        df.loc[nan_indices, col] = means.loc[
            df[nan_indices]["continent"], col
        ].to_numpy()

    # let's look at the missing values for each column again

    # except a few columns it's looking decent now, not sure what to do further with this dataset

    """
    # Other Dataset: Information about policies of countries [from Oxford University](https://github.com/OxCGRT/covid-policy-tracker)

    ### [Description of columns](https://github.com/OxCGRT/covid-policy-tracker/tree/master/documentation)
    """

    df_additional = pd.read_csv(
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv",
        parse_dates=["Date"],
    )

    # since our other dataset only has national-wide data, we'll only use it here as well
    df_additional = df_additional[df_additional["Jurisdiction"] == "NAT_TOTAL"]
    # drop columns that were used for regional description
    df_additional.drop(
        columns=["RegionName", "RegionCode", "Jurisdiction"], inplace=True
    )

    # M1_Wildcard only has missing values, lets drop that really quick
    df_additional.drop(columns=["M1_Wildcard"], inplace=True)
    # we can also drop ConfirmedCases since that's complete in out other datset anyway
    # we might be able to use ConfirmedDeaths though, because there were missing values for that in our other dataset
    df_additional.drop(columns=["ConfirmedCases"], inplace=True)

    # except for the flag attributes this looks very good, for most of them only 2% missing values
    # E3, E4, H4 is said to not be updated anymore since August 2021 on the github repo, so that's where the relatively high number of missing values comes from
    # let's check if the missing values come from the same countries

    # ok so it's well spread between countries
    # honestly this data set just looks really good, not really sure what to do
    # since the C, E, H columns are all qualitive with very few options and the Index columns result from them, I don't think it makes sense to do interpolation on these
    # let's try interpolating missing values in ConfirmedDeaths (only between first and last valid index)
    # first sort the df it by Country and then Date
    cols = df_additional.columns
    df_additional = df_additional.sort_values(by=["CountryName", "Date"]).reset_index(
        drop=True
    )
    cols_to_interpolate = ["ConfirmedDeaths"]
    df_additional = df_additional.groupby("CountryName").apply(
        lambda x: x[df_additional.columns.difference(cols_to_interpolate)].join(
            x[cols_to_interpolate].interpolate(
                method="linear", axis=0, limit_area="inside"
            )
        )
    )[cols]

    # ok that didn't do anything, but it was worth a try
    # seems like it's time to merge this dataset with the other one
    # first let's check how many countries from the our dataset are present in the additional one
    temp = [
        elem in df_additional["CountryName"].unique()
        for elem in df["location"].unique()
    ]

    # that seems high enough, so we can actually use this dataset
    # let's rename columns from the old dataset for merging
    df_additional.rename(
        columns={"CountryName": "location", "Date": "date"}, inplace=True
    )
    # we can drop redundant columns (we drop new_deaths/total_deaths, because the other dataset is more complete on these)
    df.drop(
        columns=["stringency_index", "iso_code", "new_deaths", "total_deaths"],
        inplace=True,
    )
    df = df.merge(df_additional, on=["location", "date"])
    # we have to recalculate new_deaths
    temp = (
        df.groupby("location")
        .apply(
            lambda x: np.append(
                x["ConfirmedDeaths"].iloc[1:2].to_numpy(),
                x["ConfirmedDeaths"].diff().to_numpy()[1:],
                axis=0,
            )
        )
        .to_numpy()
    )
    temp = [elem for sublist in temp for elem in sublist]
    df["new_deaths"] = temp
    df.head()

    # let's add the weekday for each row, this might be interesting to look at
    df["weekday"] = df["date"].dt.dayofweek

    # let's look at the value ranges that we have now
    df.describe()

    # let's look for some outliers
    # use rolling mean and std to identify them (add a constant term for small values)
    cols_to_check = ["new_cases", "new_deaths", "new_tests"]
    sums_of_these = ["total_cases", "ConfirmedDeaths", "total_tests"]
    rolling = (
        df[cols_to_check + ["location"]]
        .groupby("location")
        .apply(lambda x: x.rolling(30, min_periods=1).mean())
        + 1000
        + 4
        * df[["new_cases", "new_deaths", "location"]]
        .groupby("location")
        .apply(lambda x: x.rolling(30, min_periods=1).std())
    )
    temp = df[cols_to_check] > rolling

    outliers = temp[temp > 0].stack().index.tolist()
    for elem in outliers:
        # set all outliers to NaN, we'll interpolate them later
        df.loc[elem[0], elem[1]] = np.nan

    # so for some reason we have a negative number of new_cases/new_deaths/.., this is obviously an error
    # let's try to interpolate these
    # first set them to NaN
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_fix = df[numerical_cols].min(axis=0) < 0
    cols_to_fix = [
        name for name, need_fix in zip(cols_to_fix.index, cols_to_fix) if need_fix
    ]
    cols_to_fix = list(set(cols_to_fix + cols_to_check))
    # need to add location for grouping
    cols_to_fix = cols_to_fix + ["location"]
    # set negative values to NaN
    df[cols_to_fix[:-1]] = (
        df[cols_to_fix].groupby("location").apply(lambda x: x.where(x >= 0, np.nan))
    )
    # now interpolate them
    df = df.groupby("location").apply(
        lambda x: x[df.columns.difference(cols_to_fix)].join(
            x[cols_to_fix].interpolate(method="linear", axis=0, limit_area="inside")
        )
    )
    # recompute total_cases, total_deaths, total_tests
    df[sums_of_these] = (
        df[["location"] + cols_to_check]
        .groupby("location")
        .apply(lambda x: x.cumsum())[cols_to_check]
        .to_numpy()
    )

    return df
