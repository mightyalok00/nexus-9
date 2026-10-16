# NEXUS-9 FutureBench Data Dictionary

| Column | Type | Description | Early Warning Feature? |
|---|---|---|---|
| `scenario_id` | Integer | Unique identifier | ? No (Excluded identifier) |
| `environment_regime` | Categorical | Macro regime (normal, demand_surge, extreme_weather) | ? Yes |
| `economic_pressure` | Float | Macroeconomic inflation and cost pressure | ? Yes |
| `weather_severity` | Float | Severe weather score (0 to 1) | ? Yes |
| `supplier_reliability` | Float | Historical tier-1 supplier fulfillment rate (contains NaNs) | ? Yes |
| `demand_shock` | Float | Exogenous demand disturbance | ? Yes |
| `policy_strength` | Float | Regulatory and inventory buffer policy strength | ? Yes |
| `capacity` | Float | Nominal installed production capacity | ? Yes |
| `demand` | Float | Total aggregate order demand | ? Yes |
| `supplier_delay_days` | Float | Upstream component shipment delay | ? Yes |
| `available_capacity` | Float | Effective capacity after weather & reliability impact | ? Yes |
| `inventory_gap` | Float | Demand minus available capacity (floored at 0) | ? No (Outcome definition) |
| `delivery_delay_days` | Float | Downstream customer delivery delay | ? No (Outcome definition) |
| `operating_cost` | Float | Total operating expenditure (includes outlier tail) | ? Yes |
| `service_level` | Float | Customer order fulfillment percentage (0 to 100) | ? No (Outcome definition) |
| `disruption_risk` | Integer | Target: 1 = Disrupted, 0 = Normal | Target (Primary) |
