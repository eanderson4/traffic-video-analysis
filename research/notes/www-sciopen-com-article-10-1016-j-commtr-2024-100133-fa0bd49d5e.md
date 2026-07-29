# notes: https://www.sciopen.com/article/10.1016/j.commtr.2024.100133

### https://www.sciopen.com/article/10.1016/j.commtr.2024.100133

## Research Notes: "Vehicle trajectory dataset from drone videos including off-ramp and congested traffic – Analysis of data quality, traffic flow, and accident risk"

- **Publication:** *Communications in Transportation Research*, Volume 4, Issue 2, June 2024, Article number 100133.
- **DOI:** 10.1016/j.commtr.2024.100133
- **Open Access:** Yes, under CC BY license (http://creativecommons.org/licenses/by/4.0/).
- **Dates:** Received 26 January 2024, Revised 9 April 2024, Accepted 9 April 2024, Published 22 June 2024.

### Authors & Affiliations
- **Moritz Berghaus** (a) (corresponding author? – indicated by parentheses)
- **Serge Lamberty** (a)
- **Jörg Ehlers** (a)
- **Eszter Kalló** (a)
- **Markus Oeser** (a, b)
  - (a) Institute of Highway Engineering, RWTH Aachen University, Aachen, 52074, Germany
  - (b) Federal Highway Research Institute, Bergisch Gladbach, 51427, Germany

### Dataset Description
- **Location:** German highway, two lanes per direction.
- **Features:** Off-ramp and congested traffic in one direction; on-ramp in the other direction.
- **Road segment length:** ~1,200 m.
- **Duration:** 87 minutes.
- **Number of trajectories:** 8,648.
- **Situation:** Congested traffic in one direction; opposite direction includes on-ramp.

### Methodology
- **Data collection:** Drone videos.
- **Object detection:** Posttrained YOLOv5 model.
- **Projection:** Three-dimensional (3D) camera calibration to road surface.
- **Postprocessing:** Compensates for most false detections; yields accurate speeds and accelerations.
- **Validation:** Compared with induction loop data and vehicle-based smartphone sensor data.

### Data Quality (plausibility and quality estimates)
- **Speed deviation:** 0.45 m/s.
- **Acceleration deviation:** 0.3 m/s².

### Applications Mentioned
- Traffic flow analysis.
- Accident risk analysis.
- (Abstract also mentions research fields: traffic flow, traffic safety, automated driving.)

### Keywords
- Vehicle trajectory dataset
- Traffic flow
- Traffic safety
- Computer vision

### Metrics (as of retrieval date unknown)
- 3,245 Views
- 163 Downloads
- 43 Crossref citations
- 42 Web of Science citations
- 48 Scopus citations
- (Note: these metrics may change over time.)

### Licensing
- © 2024 The Authors. Open access under CC BY 4.0.

### Publisher
- Tsinghua University Press (noted in About Us section of platform).

---

**RELEVANT LEADS:**
- Institute of Highway Engineering, RWTH Aachen University, Aachen, 52074, Germany.
- Federal Highway Research Institute, Bergisch Gladbach, 51427, Germany.
- Dataset (apparently available – contact authors or check journal for data repository link).
- YOLOv5 object detection model (posttrained version used).
- Communications in Transportation Research journal.
- SciOpen platform (https://www.sciopen.com) – article hosted here.
- DOI: 10.1016/j.commtr.2024.100133.
- CC BY license text: http://creativecommons.org/licenses/by/4.0/.
- Three potential sources for validation: induction loop data, smartphone sensor data (not further specified).
