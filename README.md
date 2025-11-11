# hackathon-porject
Vision:
Wildfires often start not because of sudden sparks, but because vegetation silently dries out until ignition becomes inevitable. Our vision is to shift from reacting to fires to preventing them — by detecting critical dryness before flames appear. We aim to give communities, farmers, and environmental authorities the power to act early and save both land and lives.

Goal:
The project’s goal is to build a localized dryness risk dashboard that continuously tracks environmental data to predict when and where vegetation is reaching a critical dryness point. Instead of relying on broad seasonal forecasts (“it’s dry every summer”), the system provides real-time, area-specific alerts so decision-makers can plan clearing, patrolling, or water management more efficiently.

How It Works:
The system integrates real-time weather data (temperature, humidity, rainfall, wind, and solar radiation) from open sources such as NASA POWER and Cyprus Weather. It also incorporates satellite indicators like NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index) to evaluate vegetation greenness and leaf water content. Where possible, it will later include ground sensors (soil and leaf moisture probes) to validate local readings.

All these inputs feed into a simplified water balance model (FMCA) that estimates vegetation dryness levels. The results are visualized in a user-friendly dashboard that classifies risk as Safe, Watch, or Danger. During the hackathon, if live data connections are limited, we will simulate results using CSV datasets of historical values.
