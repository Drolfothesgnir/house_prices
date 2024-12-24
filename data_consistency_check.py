def check_garage_consistency(df):
    # First check if has_garage matches with GarageType
    garage_type_consistent = ((df["has_garage"] == 0) & (df["GarageType"].astype(str) == "NA")) | \
                             ((df["has_garage"] == 1) & (df["GarageType"].astype(str) != "NA"))

    # Then check if all garage features are consistent when garage exists
    garage_features_consistent = ((df["has_garage"] == 0) |
                                  ((df["GarageType"].astype(str) != "NA") &
                                   (df["GarageFinish"].astype(str) != "NA") &
                                   (df["GarageCars"] > 0) &  # Changed from != 0 to > 0
                                   (df["GarageArea"] > 0) &  # Changed from != 0 to > 0
                                   (df["GarageQual"].astype(str) != "NA") &
                                   (df["GarageCond"].astype(str) != "NA")))

    df["garage_type_consistent"] = garage_type_consistent
    df["garage_features_consistent"] = garage_features_consistent
    return df


def check_garage_area_reasonable(df):
    # Typical ranges: 1-car = 200-400 sq ft, 2-car = 400-600 sq ft, 3-car = 600-800 sq ft
    area_per_car_low = 180  # Minimum reasonable sq ft per car
    area_per_car_high = 400  # Maximum reasonable sq ft per car

    area_reasonable = ((df["has_garage"] == 0) |
                       ((df["GarageArea"] >= df["GarageCars"] * area_per_car_low) &
                        (df["GarageArea"] <= df["GarageCars"] * area_per_car_high)))

    df["garage_area_reasonable"] = area_reasonable
    return df


def check_basement_consistency(df):
    # Basic feature consistency check
    basement_features_consistent = ((df["has_basement"] == 0) |
                                    ((df["BsmtQual"].astype(str) != "NA") &
                                     (df["BsmtCond"].astype(str) != "NA") &
                                     (df["BsmtExposure"].astype(str) != "NA") &
                                     (df["BsmtFinType1"].astype(str) != "NA") &
                                     (df["BsmtFinSF1"] >= 0) &  # Changed from > 0 to >= 0
                                     (df["TotalBsmtSF"] > 0)))

    # Check second finished area consistency
    has_consistent_second_finished_area = ((df["BsmtFinType2"].astype(str) == "NA") &
                                           (df["BsmtFinSF2"] == 0)) | \
                                          ((df["BsmtFinType2"].astype(str) != "NA") &
                                           (df["BsmtFinSF2"] > 0))

    # Check if areas sum up correctly
    area_sums_match = (df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["BsmtUnfSF"] -
                       df["TotalBsmtSF"]).abs() < 1  # Using small tolerance for float comparison

    df["basement_features_consistent"] = basement_features_consistent
    df["has_consistent_second_finished_area"] = has_consistent_second_finished_area
    df["basement_areas_match"] = area_sums_match

    return df


def check_data_consistency(df):
    df = check_garage_consistency(df)
    df = check_garage_area_reasonable(df)
    df = check_basement_consistency(df)
    return df