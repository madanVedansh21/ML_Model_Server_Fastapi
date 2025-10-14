import numpy as np


class SCADAFeatureExtractor:
    """SCADA feature extractor mapping raw SCADA fields to model inputs.

    Expected model inputs (from metadata):
    - VLF_mean, VLF_std, temp_grad, h2_ratio, voltage_kv, power_mva, age_years

    We derive:
    - temp_grad = oil_temp_top - oil_temp_bottom
    - h2_ratio = h2_gas_ppm / max(ch4_gas_ppm, 1e-6)
    - VLF_mean ~= thd_voltage (as a proxy)
    - VLF_std ~= thd_current (as a proxy)
    Remaining fields are taken directly.
    """

    def __init__(self):
        self.expected_features = [
            "VLF_mean",
            "VLF_std",
            "temp_grad",
            "h2_ratio",
            "voltage_kv",
            "power_mva",
            "age_years",
        ]

    @staticmethod
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    def extract(self, row: dict) -> dict:
        # Derive from raw keys when present
        oil_top = self._to_float(row.get("oil_temp_top"))
        oil_bottom = self._to_float(row.get("oil_temp_bottom"))
        temp_grad = oil_top - oil_bottom

        h2_ppm = self._to_float(row.get("h2_gas_ppm"))
        ch4_ppm = self._to_float(row.get("ch4_gas_ppm"))
        denom = ch4_ppm if ch4_ppm != 0.0 else 1e-6
        h2_ratio = h2_ppm / denom

        thd_v = self._to_float(row.get("thd_voltage"))
        thd_i = self._to_float(row.get("thd_current"))

        features = {
            "VLF_mean": self._to_float(row.get("VLF_mean", thd_v)),
            "VLF_std": self._to_float(row.get("VLF_std", thd_i)),
            "temp_grad": self._to_float(row.get("temp_grad", temp_grad)),
            "h2_ratio": self._to_float(row.get("h2_ratio", h2_ratio)),
            "voltage_kv": self._to_float(row.get("voltage_kv")),
            "power_mva": self._to_float(row.get("power_mva")),
            "age_years": self._to_float(row.get("age_years")),
        }

        # Ensure all expected features exist
        for name in self.expected_features:
            if name not in features:
                features[name] = 0.0

        return features
