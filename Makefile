.PHONY: synth features train serve golden

synth:
	python etl/generate_synthetic_data.py

segments:
	python etl/stage_segmentation.py

features: segments
	python etl/build_features.py

train:
	python training/train_gbm.py

golden:
	python service/golden_curve.py

serve:
	uvicorn service.app:app --reload
