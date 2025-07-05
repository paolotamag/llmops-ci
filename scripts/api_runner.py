import os
import sys
import json
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from langfuse import Langfuse

# Load environment variables from .env file
load_dotenv()

class ExperimentRunner:
    def __init__(self):
        self.validate_environment()
        self.langfuse = Langfuse()
        
    def validate_environment(self):
        """Validate required environment variables"""
        required_vars = ['LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_HOST']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        print("✅ Environment variables validated")

    def compare_prepared_answer_ids(self, output: Dict[str, Any], expected_output: Dict[str, Any]) -> int:
        """
        Compare prepared answer IDs between output and expected output
        
        Args:
            output: Model output containing prepared_answer_id
            expected_output: Expected output containing prepared_answer_id
            
        Returns:
            1 if IDs match, 0 otherwise
        """
        if int(output.get("prepared_answer_id")) == expected_output.get("prepared_answer_id"):
            return 1
        else:
            return 0

    def run_model(self, question: str) -> Dict[str, Any]:
        """
        Placeholder for your model logic
        Replace this with your actual model implementation
        
        Args:
            question: Input question for the model
            
        Returns:
            Dict containing model output with prepared_answer_id
        """
        # TODO: Replace this with your actual model logic
        # This is a placeholder that returns a dummy response
        print(f"🤖 Running model for question: {question[:50]}...")
        
        # Import your actual model here
        try:
            from model.run_model import run_model as actual_run_model
            print("✅ Successfully imported run_model")

        except ImportError:
            print("⚠️ model.run_model not found, using placeholder")
            # Placeholder response - replace with actual model
            return {
                "prepared_answer_id": 0,
                "answer": "This is a placeholder response",
                "confidence": 0.85
            }
        
        try:
            return actual_run_model(question)
        except Exception as e:
            print(f"❌ Error running model: {e}")
            return {
                "prepared_answer_id": 0,
                "answer": "Error processing question",
                "confidence": 0.0
            }

    def run_experiment(self, experiment_name: str) -> float:
        """
        Run experiment on the pet food dataset
        
        Args:
            experiment_name: Name of the experiment run
            
        Returns:
            Success metric (success rate as float)
        """
        print(f"🧪 Starting experiment: {experiment_name}")
        
        try:
            # Get the dataset
            dataset = self.langfuse.get_dataset("pet_food_kwnown_answers")
            print(f"📊 Loaded dataset with {len(dataset.items)} items")
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            sys.exit(1)

        items_number = len(dataset.items)
        success_count = 0
        
        print(f"🏃 Processing {items_number} items...")
        
        for i, item in enumerate(dataset.items, 1):
            print(f"Processing item {i}/{items_number}")
            
            try:
                # Use the item.run() context manager
                with item.run(run_name=experiment_name) as root_span:
                    # All subsequent langfuse operations within this block are part of this trace
                    
                    # Call your application logic
                    output = self.run_model(item.input["question"])
                    
                    # Compare results
                    comparison_result = self.compare_prepared_answer_ids(output, item.expected_output)
                    success_count += comparison_result
                    
                    # Score the result against the expected output
                    root_span.score_trace(name="prepared_answer_id", value=comparison_result)
                    
                    print(f"  ✅ Item {i}: {'SUCCESS' if comparison_result else 'FAILED'}")
                    
            except Exception as e:
                print(f"  ❌ Error processing item {i}: {e}")
                continue

        # Calculate success metric
        success_metric = success_count / items_number if items_number > 0 else 0
        
        print(f"\n🎯 Finished processing dataset 'Pet Food Customer Service' for run '{experiment_name}'.")
        print(f"📈 Success rate: {success_count}/{items_number} ({(success_count/items_number)*100:.2f}%)")
        
        return success_metric

    def save_results(self, experiment_name: str, success_metric: float) -> None:
        """Save experiment results to file"""
        results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "success_rate": success_metric,
            "success_percentage": success_metric * 100
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_results_{experiment_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")

    def run(self):
        """Main execution method"""
        print("🚀 Starting Langfuse Experiment Runner...")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        
        # Get experiment name from environment or use default
        experiment_name = os.getenv('EXPERIMENT_NAME', 'test_reproducibility')
        print(f"🧪 Experiment name: {experiment_name}")
        
        try:
            # Run the experiment
            success_metric = self.run_experiment(experiment_name)
            
            # Save results
            self.save_results(experiment_name, success_metric)
            
            print("🎉 Experiment completed successfully!")
            
            # Exit with error code if success rate is below threshold
            threshold = float(os.getenv('SUCCESS_THRESHOLD', '0.8'))
            if success_metric < threshold:
                print(f"⚠️ Success rate {success_metric:.2%} below threshold {threshold:.2%}")
                sys.exit(1)
            
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            sys.exit(1)

def main():
    """Entry point"""
    runner = ExperimentRunner()
    runner.run()

if __name__ == "__main__":
    main()