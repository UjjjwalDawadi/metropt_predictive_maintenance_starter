# MetroPT Dashboard - Recalculate With Progress Version

This version restores alarm recalculation from timestamp-level model scores, but adds progress feedback so the page does not feel frozen.

## Key behaviour

The Interactive Alarm Simulator recalculates when you click Run simulation after changing:

- model score
- threshold
- persistence duration
- date range

It shows progress through:

1. loading selected timestamp range
2. recalculating alarm episodes
3. evaluating event detection and alarm burden
4. preparing timeline charts
5. completing simulation

## Important demo advice

Keep the date range small for a smooth demo, for example:

2020-07-14 to 2020-07-16

This is enough to show the final test failure event.

## Replace file

Replace:

dashboard/streamlit_app.py

with this file.

## Run

streamlit run dashboard/streamlit_app.py
