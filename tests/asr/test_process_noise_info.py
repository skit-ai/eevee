from eevee.metrics.asr import extract_info_tags, clean_info_tags, define_noisy


def test_process_noise_info():

	clean_transcription = "This is an example"
	noise_tags = ["", "['<audio_silent>']", "['<inauidble>']", "['<background_speech>']"]
	noise_labels = [0, 0, 1, 1]

	noisy_transcriptions = ["{}{}".format(clean_transcription, tag) for tag in noise_tags]

	for transcription, tag, label in zip(noisy_transcriptions, noise_tags, noise_labels):
		assert extract_info_tags(transcription) == tag
		assert clean_info_tags(transcription) == clean_transcription
		assert define_noisy(tag) == label
