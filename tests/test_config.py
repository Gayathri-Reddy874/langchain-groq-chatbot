from src.config import AppSettings


def test_missing_api_key_is_invalid():
    settings = AppSettings(groq_api_key="")
    is_valid, message = settings.is_valid()
    assert is_valid is False
    assert "API key" in message


def test_valid_settings_pass():
    settings = AppSettings(groq_api_key="fake-key", temperature=0.5)
    is_valid, message = settings.is_valid()
    assert is_valid is True
    assert message == ""


def test_temperature_out_of_range_is_invalid():
    settings = AppSettings(groq_api_key="fake-key", temperature=1.5)
    is_valid, message = settings.is_valid()
    assert is_valid is False
    assert "Temperature" in message


def test_unknown_model_is_invalid():
    settings = AppSettings(groq_api_key="fake-key", model="not-a-real-model")
    is_valid, message = settings.is_valid()
    assert is_valid is False
    assert "Unknown model" in message

