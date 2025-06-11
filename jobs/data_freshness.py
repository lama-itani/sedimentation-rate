"""
Data validation and freshness checking job.
Handles data quality checks, schema validation, and freshness verification.
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

class DataValidationJob:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check if new data is available for retraining"""
        data_path = self.config.get("data_path", "../data/raw/Data_set_cas_1.csv")
        
        try:
            if not os.path.exists(data_path):
                self.logger.error(f"🚨 Data file not found: {data_path}")
                return {
                    "is_fresh": False,
                    "reason": "file_not_found",
                    "path": data_path
                }
            
            # Check file modification time
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
            freshness_threshold = datetime.now() - timedelta(
                hours=self.config.get("freshness_hours", 24)
            )
            
            is_fresh = file_mod_time > freshness_threshold
            
            result = {
                "is_fresh": is_fresh,
                "file_modified": file_mod_time.isoformat(),
                "threshold": freshness_threshold.isoformat(),
                "path": data_path
            }
            
            if is_fresh:
                self.logger.info("✅ Data is fresh and ready for training")
            else:
                self.logger.warning(f"⚠️ Data is stale. Last modified: {file_mod_time}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"🚨 Error checking data freshness: {str(e)}")
            return {
                "is_fresh": False,
                "reason": "check_failed",
                "error": str(e)
            }
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Perform data quality checks"""
        data_path = self.config.get("data_path", "data/raw/Data_set_cas_1.csv")
        
        try:
            # Load data for quality checks
            df = pd.read_csv(data_path, sep=";")
            
            quality_metrics = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                "duplicate_rows": df.duplicated().sum(),
                "target_missing": df['TARGET'].isnull().sum() if 'TARGET' in df.columns else 0
            }
            
            # Quality thresholds
            max_missing_pct = self.config.get("max_missing_percentage", 30)
            min_rows = self.config.get("min_rows", 50)
            
            # Validation checks
            quality_checks = {
                "sufficient_data": quality_metrics["total_rows"] >= min_rows,
                "acceptable_missing": quality_metrics["missing_percentage"] <= max_missing_pct,
                "has_target": 'TARGET' in df.columns,
                "target_not_empty": quality_metrics["target_missing"] < len(df) * 0.8
            }
            
            all_checks_passed = all(quality_checks.values())
            
            self.logger.info(f"📊 Data quality metrics: {quality_metrics}")
            
            if all_checks_passed:
                self.logger.info("✅ All data quality checks passed")
            else:
                failed_checks = [k for k, v in quality_checks.items() if not v]
                self.logger.warning(f"⚠️ Quality checks failed: {failed_checks}")
            
            return {
                "quality_passed": all_checks_passed,
                "metrics": quality_metrics,
                "checks": quality_checks,
                "failed_checks": [k for k, v in quality_checks.items() if not v]
            }
            
        except Exception as e:
            self.logger.error(f"🚨 Data quality validation failed: {str(e)}")
            return {
                "quality_passed": False,
                "error": str(e)
            }
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete data validation pipeline"""
        self.logger.info("🔷 Starting data validation job")
        
        # Check data freshness
        freshness_result = self.check_data_freshness()
        
        # Check data quality
        quality_result = self.validate_data_quality()
        
        # Overall validation result
        validation_passed = (
            freshness_result.get("is_fresh", False) and 
            quality_result.get("quality_passed", False)
        )
        
        result = {
            "validation_passed": validation_passed,
            "freshness": freshness_result,
            "quality": quality_result,
            "timestamp": datetime.now().isoformat()
        }
        
        if validation_passed:
            self.logger.info("✅ Data validation completed successfully")
        else:
            self.logger.warning("⚠️ Data validation failed")
        
        return result

if __name__ == "__main__":
    # Configuration
    config = {
        "data_path": "data/raw/Data_set_cas_1.csv",
        "freshness_hours": 24,
        "max_missing_percentage": 30,
        "min_rows": 50
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run validation
    validator = DataValidationJob(config)
    result = validator.run_validation()
    
    # Exit with appropriate code
    exit_code = 0 if result["validation_passed"] else 1
    exit(exit_code)