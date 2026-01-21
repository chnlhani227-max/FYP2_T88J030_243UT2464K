import numpy as np

class EdgeProfiler:
    HW_SPECS = {
        "Raspberry Pi 4": {"GOPS": 13.5, "Watts": 4.0},
        "Raspberry Pi 5": {"GOPS": 45.0, "Watts": 8.0},
        "Nvidia Jetson Nano": {"GOPS": 472.0, "Watts": 10.0},
        "Nvidia Jetson Orin": {"GOPS": 40000.0, "Watts": 15.0},
        "ESP32-S3": {"GOPS": 0.24, "Watts": 0.5},
        "Arduino Nano BLE": {"GOPS": 0.1, "Watts": 0.2},
        "Google Coral TPU": {"GOPS": 4000.0, "Watts": 2.0}
    }
    
    def get_device_list(self):
        return list(self.HW_SPECS.keys())
    
    def profile_model(self, model, model_type, input_shape, target_device="Raspberry Pi 4"):
        flops = 0
        params = 0
        
        if "sklearn" in model_type:
            if hasattr(model, "estimators_"):
                flops = len(model.estimators_) * 100
            else:
                flops = input_shape[1] * 2
                params = input_shape[1]
        elif model_type in ["dnn", "cnn", "rnn"]:
            try:
                params = model.count_params()
                flops = params * 2
            except:
                params = 10000
                flops = 20000
        
        specs = self.HW_SPECS.get(target_device, self.HW_SPECS["Raspberry Pi 4"])
        overhead = 2.0 if "ESP32" in target_device else 1.2
        latency_s = (flops / (specs["GOPS"] * 1e9)) * overhead
        latency_ms = max(0.1, latency_s * 1000)
        energy_mj = specs["Watts"] * (latency_ms / 1000) * 1000
        
        return {
            "Target Device": target_device,
            "Latency (ms)": round(latency_ms, 2),
            "Energy (mJ)": round(energy_mj, 2),
            "FLOPs": flops,
            "Params": params
        }
