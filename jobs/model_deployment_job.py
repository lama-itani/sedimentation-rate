"""
This job is for model deployment and registration.
It handles MLflow logging, model registration, and auto-deployment decisions.
"""
import logging
from typing import Dict, Any
from datetime import datetime

class ModelDeploymentJob:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.performance_threshold = config.get("performance_threshold", 0.45)
        
    def fetch_model_from_cloudera(self, model_name: str, data_version: str) -> Dict[str, Any]:
        """Fetch model ID from Cloudera AI Model Registry"""
        try:
            # /!\ Uncomment code when in Cloudera environment, use CML SDK:
            # import cml.models_v1 as models
            # model_registry = models.ModelsApi()
            # model_list = model_registry.list_models(search_filter=f"name:{model_name}")
            
            # Mock Cloudera AI model registry lookup
            self.logger.info(f"🔷 Fetching {model_name} model from Cloudera registry")
            
            # Simulate finding the model in registry
            model_id = f"cloudera-{model_name.lower()}-{data_version[:8]}"
            
            self.logger.info(f"✅ Model found in Cloudera registry: {model_id}")
            
            return {
                "model_id": model_id,
                "model_name": model_name,
                "data_version": data_version,
                "registry_status": "found",
                "source": "cloudera_registry"
            }
            
        except Exception as e:
            self.logger.error(f"🚨 Model lookup failed: {str(e)}")
            return {
                "registry_status": "not_found",
                "error": str(e)
            }
    
    def register_model_in_cloudera(self, model_data: Dict[str, Any], data_version: str) -> Dict[str, Any]:
        """Register new model in Cloudera AI (if training job ran successfully)"""
        try:
            best_model_info = model_data["best_model"]
            
            # Check if we have a trained pipeline
            if best_model_info.get("pipeline") is None:
                self.logger.info("🔷 No pipeline object - fetching existing model from registry")
                return self.fetch_model_from_cloudera(best_model_info["name"], data_version)
            
            # If we have a pipeline, register it (training job ran)
            import sys
            sys.path.append(self.config.get("src_path", "src"))
            from mlflow_utils import log_model
            
            self.logger.info(f"🔷 Registering new {best_model_info['name']} model")
            
            # Use actual training data if available
            test_data_shape = model_data.get("test_data_shape", (15, 61))
            import pandas as pd
            dummy_X_test = pd.DataFrame([[0] * test_data_shape[1]] * 2)
            
            model_id = log_model(
                pipeline=best_model_info["pipeline"],
                model_name=best_model_info["name"],
                metrics=best_model_info["metrics"],
                X_test=dummy_X_test,
                data_version=data_version
            )
            
            self.logger.info(f"✅ New model registered: {model_id}")
            
            return {
                "model_id": model_id,
                "model_name": best_model_info["name"],
                "registration_status": "success",
                "source": "new_training"
            }
            
        except Exception as e:
            self.logger.error(f"🚨 Model registration failed: {str(e)}")
            return {
                "registration_status": "failed",
                "error": str(e)
            }
    
    def evaluate_deployment_criteria(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if model should be auto-deployed"""
        best_score = model_data["best_model"]["score"]
        
        deployment_decision = {
            "should_deploy": best_score >= self.performance_threshold,
            "score": best_score,
            "threshold": self.performance_threshold,
            "reason": ""
        }
        
        if deployment_decision["should_deploy"]:
            deployment_decision["reason"] = f"Score {best_score:.3f} meets threshold {self.performance_threshold}"
            self.logger.info(f"✅ Model approved for deployment: {deployment_decision['reason']}")
        else:
            deployment_decision["reason"] = f"Score {best_score:.3f} below threshold {self.performance_threshold}"
            self.logger.warning(f"⚠️ Model held for review: {deployment_decision['reason']}")
        
        return deployment_decision
    
    def deploy_model_to_cloudera(self, model_id: str, model_name: str) -> Dict[str, Any]:
        """Deploy model to Cloudera AI production endpoint"""
        try:
            # In real Cloudera environment:
            # import cml
            # model = cml.get_model(model_id)
            # endpoint = model.deploy(cpu=1, memory=2, replicas=1)
            
            self.logger.info(f"🚀 Deploying {model_name} to Cloudera AI (ID: {model_id})")
            
            # Mock Cloudera deployment
            endpoint_name = f"sediment-prediction-{model_name.lower()}"
            endpoint_url = f"https://ml.cloudera.company.com/{endpoint_name}/predict"
            
            self.logger.info(f"✅ Model deployed to Cloudera endpoint: {endpoint_url}")
            
            return {
                "deployment_status": "deployed",
                "endpoint_url": endpoint_url,
                "endpoint_name": endpoint_name,
                "model_id": model_id,
                "platform": "cloudera_ai",
                "deployed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"🚨 Cloudera deployment failed: {str(e)}")
            return {
                "deployment_status": "failed",
                "error": str(e)
            }
    
    def run_deployment(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete deployment pipeline for Cloudera AI"""
        self.logger.info("🔷 Starting model deployment job")
        
        if training_results.get("status") != "success":
            return {
                "deployment_status": "skipped",
                "reason": "training_failed"
            }
        
        try:
            # Register or fetch model from Cloudera AI
            registration_result = self.register_model_in_cloudera(
                training_results, 
                training_results["data_version"]
            )
            
            if registration_result.get("registry_status") == "not_found" and registration_result.get("registration_status") == "failed":
                return {
                    "deployment_status": "failed",
                    "reason": "model_not_available",
                    "error": registration_result.get("error")
                }
            
            # Evaluate deployment criteria
            deployment_decision = self.evaluate_deployment_criteria(training_results)
            
            # Deploy if criteria met
            if deployment_decision["should_deploy"]:
                deployment_result = self.deploy_model_to_cloudera(
                    registration_result["model_id"],
                    registration_result["model_name"]
                )
            else:
                deployment_result = {
                    "deployment_status": "held_for_review",
                    "reason": deployment_decision["reason"],
                    "platform": "cloudera_ai"
                }
            
            # Combine results
            final_result = {
                **registration_result,
                **deployment_result,
                "deployment_decision": deployment_decision,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("✅ Deployment job completed")
            return final_result
            
        except Exception as e:
            self.logger.error(f"🚨 Deployment job failed: {str(e)}")
            return {
                "deployment_status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

if __name__ == "__main__":
    # Configuration
    config = {
        "src_path": "src",
        "performance_threshold": 0.45
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Mock training results for testing
    mock_training_results = {
        "status": "success",
        "best_model": {
            "name": "Lasso",
            "score": 0.505,
            "pipeline": None,  # Would be actual pipeline
            "metrics": {"test_r2": 0.505}
        },
        "data_version": "abc123",
        "test_data_shape": (15, 61)
    }
    
    # Run deployment
    deployer = ModelDeploymentJob(config)
    result = deployer.run_deployment(mock_training_results)
    
    print(f"Deployment result: {result}")
    
    # Exit with appropriate code
    exit_code = 0 if result.get("deployment_status") in ["deployed", "held_for_review"] else 1
    exit(exit_code)