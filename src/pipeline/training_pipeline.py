from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:

    def __init__(self):

        self.raw_data_path = (
            "data/raw/"
            "energydata_complete.csv"
        )

        self.transformed_data_path = (
            "data/processed/"
            "transformed_energy_data.csv"
        )

        self.model_path = (
            "models/"
            "gradient_boosting_energy_model.pkl"
        )

    def run_pipeline(self):

        print("\n")
        print("=" * 60)
        print("       ENERGY CONSUMPTION ML PIPELINE")
        print("=" * 60)

        # =========================================
        # STEP 1: DATA INGESTION
        # =========================================

        print("\n[1/5] DATA INGESTION")
        print("-" * 40)

        ingestion = DataIngestion(
            self.raw_data_path
        )

        ingestion.initiate_data_ingestion()

        # =========================================
        # STEP 2: DATA VALIDATION
        # =========================================

        print("\n[2/5] DATA VALIDATION")
        print("-" * 40)

        validation = DataValidation(
            self.raw_data_path
        )

        validation_result = (
            validation.validate_data()
        )

        if not validation_result:

            print(
                "\nData validation failed."
            )

            print(
                "Pipeline stopped."
            )

            return

        # =========================================
        # STEP 3: DATA TRANSFORMATION
        # =========================================

        print("\n[3/5] DATA TRANSFORMATION")
        print("-" * 40)

        transformation = DataTransformation(
            self.raw_data_path
        )

        transformation.initiate_data_transformation()

        # =========================================
        # STEP 4: MODEL TRAINING
        # =========================================

        print("\n[4/5] MODEL TRAINING")
        print("-" * 40)

        trainer = ModelTrainer(
            self.transformed_data_path
        )

        trainer.initiate_model_training()

        # =========================================
        # STEP 5: MODEL EVALUATION
        # =========================================

        print("\n[5/5] MODEL EVALUATION")
        print("-" * 40)

        evaluator = ModelEvaluation(
            self.model_path,
            self.transformed_data_path
        )

        evaluator.initiate_model_evaluation()

        # =========================================
        # PIPELINE COMPLETE
        # =========================================

        print("\n")
        print("=" * 60)
        print("       PIPELINE EXECUTION COMPLETED")
        print("=" * 60)


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run_pipeline()