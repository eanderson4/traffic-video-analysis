# notes: https://pmc.ncbi.nlm.nih.gov/articles/PMC12241643/

### https://pmc.ncbi.nlm.nih.gov/articles/PMC12241643/

## Publication Info
- **Journal**: *Scientific Data* (Sci Data), 2025 Jul 9;12:1174  
- **DOI**: 10.1038/s41597-025-05472-0  
- **PMCID**: PMC12241643 | **PMID**: 40634368  
- **Received**: 2025 Mar 12 | **Accepted**: 2025 Jun 25 | **Collection date**: 2025  
- **License**: Creative Commons Attribution 4.0 International  

## Dataset Overview
- **Name**: MiTra (Milan Trajectories)  
- **Description**: Drone-based high-resolution traffic trajectory dataset covering a 900 m section of the A50 urban freeway in Milan, Italy (Rozzano District).  
- **Road features**: Six lanes (plus additional lanes for ramp connections), four ramps (two on-ramps, two off-ramps).  
- **Collection date**: April 27, 2023 (rescheduled from Apr 13 and Apr 20 due to rain).  
- **Weather**: 18 °C, wind 13 km/h.  

## Collection Details
- **Drones**: Six DJI Mini 2; 4K resolution (4096×2160) at 30 fps; flown at height 120 m (EASA compliant).  
- **Flight campaigns**: Nine flights, average duration 15 min each.  
- **Total recording**: 135 min net (over 3.5 h, 15:15–18:55). Each drone recorded 13.5 h of footage across all flights.  
- **Flight team**: Seven pilots + one central coordinator; operated by “Immagini al volo”.  
- **Take‑off/landing**: Drones 1‑3 from Location 1; Drones 4‑6 from Location 2.  
- **Synchronization**: All drones started/stopped recording simultaneously via central coordinator.  
- **Data extraction**: Outsourced to DataFromSky company; used Multiple Hypothesis Tracking (MHT) + Kalman filter.  
- **Video stitching**: Five transitions; georegistered to UTM coordinate system.  
- **Privacy**: Top‑down view (no license plates or faces); compliant with EASA/ENAC/ENAV regulations.  

## Data Characteristics
- **Total trajectories**: 124,641 (single‑drone) + 24,161 (stitched, all six drones).  
- **Total distance covered**: >20,000 km.  
- **Frame rate**: 30 Hz.  
- **Vehicle types**: Cars (73%), Medium Vehicles (13.4%), Heavy Vehicles (11.3%), Motorcycles (2.1%), Buses (0.2%).  
- **Movement**: 76.9% straight, 13% merged (on‑ramp), 10.1% diverged (off‑ramp).  
- **Lane changes**: 51.9% of vehicles changed lanes; 24.8% single change, 27.1% multiple changes.  
- **By type**: Cars 52.8%, Medium 49.3%, Heavy 42.3%, Motorcycles 85.7%.  

### Data Attributes (CSV columns)
- **Vehicle_ID**, **Vehicle_type**, **Time [s]**, **x [m]** (UTM longitude), **y [m]** (UTM latitude), **Speed [km/h]**, **Lon. Acc. [m/s²]**, **Lat. Acc. [m/s²]**, **Angle [rad]**, **Vehicle_length [m]**, **Vehicle_width [m]**, **Lane** (0‑3 mainline L→R; 4‑7 mainline R→L; 10‑11 ramps L→R; 20‑21 ramps R→L), **Leader_ID**, **Follower_ID**, **Left_Leader_ID**, **Left_Follower_ID**, **Right_Leader_ID**, **Right_Follower_ID**.  

## Quality & Validation
- **Detection/tracking**: Manual cross‑check by DataFromSky; no ID/tracking errors reported.  
- **Global positioning error**: 25.3 cm (average distance between 13 lamp posts vs. extracted coordinates).  
- **Stitching lateral shift**: ~0.5 m at 320 m and 580 m due to drones flown ~130 m to side (EASA restriction); correctable.  
- **Plausibility checks**:  
  - Single‑drone data: 26 longitudinal jumps (>2 m/step → 60 m/s) by 3 vehicles; 914 lateral jumps (>0.5 m/step → 15 m/s) by 595 vehicles (out of 124,641 vehicles, 107,663,239 timestamps).  
  - Stitched data: 27 longitudinal jumps by 3 vehicles; 253 lateral jumps by 60 vehicles (out of 24,161 vehicles, 63,780,893 timestamps).  
- **Traffic states**: Free‑flow (Flight 3), dense (Flight 4), congested with stop‑and‑go (Flight 8).  
- **Wave speeds**: ~4.3 m/s (dense), ~4 m/s (congested), ~4.6 m/s overall flow‑density.  
- **Speed‑flow‑density**: Follows classical theoretical patterns; Edie’s method (100 m stretch, 20 s window).  

## Access & Usage
- **Repository**: OPARA (Open Access Repository of Saxon Universities)  
  **DOI**: 10.25532/OPARA-881  
- **Contents**: Raw videos (MP4), tracking logs (tlgx/ftlgx), extracted trajectory CSV files.  
- **Folder structure**: `Data_T[flight#]` → CSV (e.g., `T1_D1.csv`); `Tracking_Logs_Videos_T[flight#]` → video and tracking files.  
- **Viewer**: DataFromSky Viewer software at `https://www.datafromsky.com/download/DataFromSkyViewer.exe`  
- **Sample code**: Python script at `https://github.com/ankitiitm/MiTra`  
- **Important**: Flights are temporally discontinuous; each flight should be analyzed independently.  
- **Potential uses**: Car‑following models, lane‑changing/integrated models, autonomous vehicle algorithms, urban planning, safety analysis.  

## Authors & Funding
- **Authors**: Ankit Anil Chaudhari (corresponding), Martin Treiber, Ostap Okhrin  
- **Affiliations**:  
  1. Faculty of Transport and Traffic Sciences, Technische Universität Dresden, Dresden, Germany  
  2. Center for Scalable Data Analytics and Artificial Intelligence (ScaDS.AI), Dresden/Leipzig, Germany  
- **Funding**: DFG project grant no. 456691906 – “Enhancing Traffic Flow Understanding by Two‑Dimensional Microscopic Models – ETF2D”.  
- **Acknowledgments**: DataFromSky team, Immagini al volo (UAV pilots), TU Dresden ZIH (high‑performance computing).  
- **Competing interests**: None declared.

## Relevant Leads
- **Dataset DOI**: 10.25532/OPARA-881  
- **DataFromSky**: https://datafromsky.com/  
- **DataFromSky Viewer download**: https://www.datafromsky.com/download/DataFromSkyViewer.exe  
- **GitHub sample code**: https://github.com/ankitiitm/MiTra  
- **DFG project**: 456691906 (ETF2D)  
- **EASA (drone regulations)**: https://www.easa.europa.eu/  
- **ENAC (Italian aviation authority)**: https://www.enac.gov.it/  
- **ENAV (Italian air navigation)**: https://www.enav.it/  
- **OPARA repository**: https://opara.zih.tu-dresden.de/ (implied)  
- **Reference 29 (MiTra dataset)**: Chaudhari et al. (2025), DOI 10.25532/OPARA-881
