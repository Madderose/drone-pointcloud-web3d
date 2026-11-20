# Disclaimers & Compliance Notices

## 1. Drone Flight & Airspace Regulations

You are solely responsible for planning and conducting any Unmanned Aircraft System (UAS / drone) flights in strict compliance with all applicable national and local aviation regulations, including (but not limited to):
- **European Union (EASA / DGAC):** Open/Specific category rules, remote pilot competency certificates, UAS operator registration, remote ID, VLOS (Visual Line of Sight) requirements, altitude ceilings, and national geofencing maps (e.g., Géoportail / AlphaTango in France).
- **United States (FAA):** Part 107 / TRUST certification, airspace authorizations (LAANC), and altitude/VLOS restrictions.
- **Other Jurisdictions:** All applicable civil aviation authority rules, no-fly zones, protected natural reserves, and critical infrastructure buffers.

This repository provides post-processing software only and does not verify flight legality or airspace authorization.

---

## 2. Privacy, Property & Bystander Consent

Aerial imagery and 3D reconstructions may capture private property, neighboring land, vehicles, license plates, or identifiable individuals. Before recording, processing, or publishing point clouds or web viewers publicly:
- Obtain all required permissions from landowners and stakeholders.
- Comply with applicable data protection and privacy laws (such as **GDPR** in Europe), including masking, cropping, or omitting neighboring properties and bystanders where required.
- Never publish raw imagery, GPS coordinates, or 3D models of private sites without explicit authorization.

---

## 3. Surveying, Engineering & Metric Accuracy Disclaimer

This pipeline reconstructs metric 3D geometry in a local **East-North-Up (ENU)** tangent plane aligned via rigid 7-DoF similarity transformation (`Sim(3)`) to consumer drone GPS/barometric telemetry (`.SRT` and EXIF):
- **Consumer GPS Absolute Offset:** Non-RTK consumer drone GNSS typically exhibits **~0.5 m to 1.5 m** horizontal/vertical absolute positioning offset (and barometric altitude drift between flights).
- **Local Relative Geometry:** Within a well-captured multi-view orbit, relative local surface geometry typically achieves **~5 cm to 10 cm** fidelity (and sub-10 cm co-registration when locked to a reference sparse reconstruction).
- **NOT a Legal Survey or Cadastre:** Output point clouds, meshes, cross-sections, and WebGL measurements are **not** certified cadastral surveys, boundary determinations, or licensed land-surveyor / structural-engineering deliverables. Do **not** use them for legal property boundaries, structural load design, or regulatory permit filings without independent verification by a licensed surveyor or qualified engineer (or calibrated Ground Control Points / RTK-PPK).

---

## 4. Software Warranty Disclaimer

As stated in the [`LICENSE`](LICENSE) (MIT License), the software is provided **"AS IS"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from the use of this software or any 3D reconstructions derived from it.

---

## 5. OpenMVS AGPL-3.0 Notice

Stage 3 of the pipeline invokes user-installed external **OpenMVS** CLI binaries licensed under **GNU AGPL-3.0**. Refer to [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before embedding or exposing the reconstruction pipeline in any network-accessible server or SaaS workflow.
