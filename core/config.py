"""Environment configuration; credentials never belong in source control."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    GEMINI_API_KEY: str = Field(default="", repr=False)
    GEMINI_API_KEYS: str = Field(default="", repr=False)
    GLOBAL_LLM_MODEL: str = "gemini-3.8-flash"
    TOTAL_ACCOUNT_BALANCE: float = Field(10000., gt=0)
    MAX_RISK_PER_TRADE_PERCENT: float = Field(1.5, gt=0, le=100)
    MAX_LEVERAGE: int = Field(10, ge=1, le=10)
    DB_PATH: str = "quant_sentinel.db"
    HOLD_SECONDS: int = Field(2700, gt=0)
    MAX_QUOTE_GAP_SECONDS: float = Field(10., gt=0)
    TAKER_FEE: float = Field(.0004, ge=0, lt=1)
    @property
    def api_keys_list(self):
        return list(dict.fromkeys(k.strip() for k in
            (self.GEMINI_API_KEYS + "," + self.GEMINI_API_KEY).split(",")
            if k.strip() and not k.strip().startswith("YOUR_")))

settings = AppConfig()
