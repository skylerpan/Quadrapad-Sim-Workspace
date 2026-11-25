CycloneDDS XML files for the container go here; this directory is mounted read-only at `/dds`.
Select one in `../.env` with `CYCLONEDDS_URI=file:///dds/<file>.xml` (or `QUADSIM_ARGS=--dds-config /dds/<file>.xml`).
The `<Domain Id>` in the file must match `ROS_DOMAIN_ID` (default 1), or Cyclone ignores the block; the launcher warns.
