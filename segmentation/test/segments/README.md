# Segments

Single cell image of segmented honeycomb pictures from new images are written into directories here.
The names of the directories are the same as the honeycomb image name.
The name of individual cell images are the same as the segment name (`segment.name + '.png'`).

I labeled them by hand by just moving the files into subdirectories
_capped_, _nectar_, _small_larva_ and so on.
`python3 test/segments/process_segments.py` collects them all in [test/segments/all/](./all/).
The results are copied over to the _bee classes_ repository where they are used for training.
