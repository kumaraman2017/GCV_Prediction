import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.config import PROXIMATE_SUM_TOLERANCE


class CoalRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    moisture: float = Field(alias="Moisture")
    volatile_matter: float = Field(alias="Volatile_matter")
    fixed_carbon: float = Field(alias="Fixed_Carbon")
    std_ash: float = Field(alias="Std.Ash")
    gcv: float = Field(alias="GCV")

    @field_validator("moisture", "volatile_matter", "fixed_carbon", "std_ash", "gcv")
    @classmethod
    def must_be_non_negative(cls, value: float, info) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {value}")
        return value

    @field_validator("moisture", "volatile_matter", "fixed_carbon", "std_ash")
    @classmethod
    def must_be_at_most_100(cls, value: float, info) -> float:
        if value > 100:
            raise ValueError(f"{info.field_name} must be <= 100, got {value}")
        return value

    @model_validator(mode="after")
    def proximate_components_sum_to_100(self) -> "CoalRecord":
        total = self.moisture + self.volatile_matter + self.fixed_carbon + self.std_ash
        if abs(total - 100) > PROXIMATE_SUM_TOLERANCE:
            raise ValueError(
                f"Moisture+Volatile_matter+Fixed_Carbon+Std.Ash = {total:.2f}, expected ~100"
            )
        return self


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    valid_indices = []
    errors = []
    for idx, row in df.iterrows():
        try:
            CoalRecord.model_validate(row.to_dict())
            valid_indices.append(idx)
        except ValidationError as exc:
            errors.append({"index": int(idx), "errors": exc.errors()})
    return df.loc[valid_indices].reset_index(drop=True), errors
