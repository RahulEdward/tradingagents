#!/usr/bin/env python3
"""
MongoDB Report Manager
Used for saving and reading analysis reports to MongoDB database
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("pymongo not installed, MongoDB functionality unavailable")


class MongoDBReportManager:
    """MongoDB Report Manager"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
        
        if MONGODB_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv()

            # Get MongoDB configuration from environment variables
            mongodb_host = os.getenv("MONGODB_HOST", "localhost")
            mongodb_port = int(os.getenv("MONGODB_PORT", "27017"))
            mongodb_username = os.getenv("MONGODB_USERNAME", "")
            mongodb_password = os.getenv("MONGODB_PASSWORD", "")
            mongodb_database = os.getenv("MONGODB_DATABASE", "tradingagents")
            mongodb_auth_source = os.getenv("MONGODB_AUTH_SOURCE", "admin")

            logger.info(f"🔧 MongoDB configuration: host={mongodb_host}, port={mongodb_port}, db={mongodb_database}")
            logger.info(f"🔧 Authentication info: username={mongodb_username}, auth_source={mongodb_auth_source}")

            # Build connection parameters
            connect_kwargs = {
                "host": mongodb_host,
                "port": mongodb_port,
                "serverSelectionTimeoutMS": 5000,
                "connectTimeoutMS": 5000
            }

            # If username and password exist, add authentication info
            if mongodb_username and mongodb_password:
                connect_kwargs.update({
                    "username": mongodb_username,
                    "password": mongodb_password,
                    "authSource": mongodb_auth_source
                })

            # Connect to MongoDB
            self.client = MongoClient(**connect_kwargs)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Select database and collection
            self.db = self.client[mongodb_database]
            self.collection = self.db["analysis_reports"]
            
            # Create indexes
            self._create_indexes()
            
            self.connected = True
            logger.info(f"✅ MongoDB connection successful: {mongodb_database}.analysis_reports")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.connected = False
    
    def _create_indexes(self):
        """Create indexes to improve query performance"""
        try:
            # Create compound index
            self.collection.create_index([
                ("stock_symbol", 1),
                ("analysis_date", -1),
                ("timestamp", -1)
            ])
            
            # Create single field indexes
            self.collection.create_index("analysis_id")
            self.collection.create_index("status")
            
            logger.info("✅ MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ MongoDB index creation failed: {e}")
    
    def save_analysis_report(self, stock_symbol: str, analysis_results: Dict[str, Any],
                           reports: Dict[str, str]) -> bool:
        """Save analysis report to MongoDB"""
        if not self.connected:
            logger.warning("MongoDB not connected, skipping save")
            return False

        try:
            # Generate analysis ID
            timestamp = datetime.now()
            analysis_id = f"{stock_symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

            # Build document
            document = {
                "analysis_id": analysis_id,
                "stock_symbol": stock_symbol,
                "analysis_date": timestamp.strftime('%Y-%m-%d'),
                "timestamp": timestamp,
                "status": "completed",
                "source": "mongodb",

                # Analysis results summary
                "summary": analysis_results.get("summary", ""),
                "analysts": analysis_results.get("analysts", []),
                "research_depth": analysis_results.get("research_depth", 1),  # Fixed: get actual research depth from analysis results

                # Report content
                "reports": reports,

                # Metadata
                "created_at": timestamp,
                "updated_at": timestamp
            }
            
            # Insert document
            result = self.collection.insert_one(document)
            
            if result.inserted_id:
                logger.info(f"✅ Analysis report saved to MongoDB: {analysis_id}")
                return True
            else:
                logger.error("❌ MongoDB insertion failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to save analysis report to MongoDB: {e}")
            return False
    
    def get_analysis_reports(self, limit: int = 100, stock_symbol: str = None,
                           start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get analysis reports from MongoDB"""
        if not self.connected:
            return []
        
        try:
            # Build query conditions
            query = {}
            
            if stock_symbol:
                query["stock_symbol"] = stock_symbol
            
            if start_date or end_date:
                date_query = {}
                if start_date:
                    date_query["$gte"] = start_date
                if end_date:
                    date_query["$lte"] = end_date
                query["analysis_date"] = date_query
            
            # Query data
            cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)
            
            results = []
            for doc in cursor:
                # Handle timestamp field, compatible with different data types
                timestamp_value = doc.get("timestamp")
                if hasattr(timestamp_value, 'timestamp'):
                    # datetime object
                    timestamp = timestamp_value.timestamp()
                elif isinstance(timestamp_value, (int, float)):
                    # Already a timestamp
                    timestamp = float(timestamp_value)
                else:
                    # Other cases, use current time
                    from datetime import datetime
                    timestamp = datetime.now().timestamp()
                
                # Convert to format expected by web application
                result = {
                    "analysis_id": doc["analysis_id"],
                    "timestamp": timestamp,
                    "stock_symbol": doc["stock_symbol"],
                    "analysts": doc.get("analysts", []),
                    "research_depth": doc.get("research_depth", 0),
                    "status": doc.get("status", "completed"),
                    "summary": doc.get("summary", ""),
                    "performance": {},
                    "tags": [],
                    "is_favorite": False,
                    "reports": doc.get("reports", {}),
                    "source": "mongodb"
                }
                results.append(result)
            
            logger.info(f"✅ Retrieved {len(results)} analysis reports from MongoDB")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get analysis reports from MongoDB: {e}")
            return []
    
    def get_report_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get single analysis report by ID"""
        if not self.connected:
            return None
        
        try:
            doc = self.collection.find_one({"analysis_id": analysis_id})
            
            if doc:
                # Convert to format expected by web application
                result = {
                    "analysis_id": doc["analysis_id"],
                    "timestamp": doc["timestamp"].timestamp(),
                    "stock_symbol": doc["stock_symbol"],
                    "analysts": doc.get("analysts", []),
                    "research_depth": doc.get("research_depth", 0),
                    "status": doc.get("status", "completed"),
                    "summary": doc.get("summary", ""),
                    "performance": {},
                    "tags": [],
                    "is_favorite": False,
                    "reports": doc.get("reports", {}),
                    "source": "mongodb"
                }
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get report from MongoDB: {e}")
            return None
    
    def delete_report(self, analysis_id: str) -> bool:
        """Delete analysis report"""
        if not self.connected:
            return False
        
        try:
            result = self.collection.delete_one({"analysis_id": analysis_id})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Analysis report deleted: {analysis_id}")
                return True
            else:
                logger.warning(f"⚠️ Report to delete not found: {analysis_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete analysis report: {e}")
            return False

    def get_all_reports(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all analysis reports"""
        if not self.connected:
            return []

        try:
            # Get all reports, sorted by timestamp in descending order
            cursor = self.collection.find().sort("timestamp", -1).limit(limit)
            reports = list(cursor)

            # Convert ObjectId to string
            for report in reports:
                if '_id' in report:
                    report['_id'] = str(report['_id'])

            logger.info(f"✅ Retrieved {len(reports)} analysis reports from MongoDB")
            return reports

        except Exception as e:
            logger.error(f"❌ Failed to get all reports from MongoDB: {e}")
            return []

    def fix_inconsistent_reports(self) -> bool:
        """Fix inconsistent report data structure"""
        if not self.connected:
            logger.warning("MongoDB not connected, skipping fix")
            return False

        try:
            # Find documents missing reports field or with empty reports field
            query = {
                "$or": [
                    {"reports": {"$exists": False}},
                    {"reports": {}},
                    {"reports": None}
                ]
            }

            cursor = self.collection.find(query)
            inconsistent_docs = list(cursor)

            if not inconsistent_docs:
                logger.info("✅ All report data structures are consistent, no fix needed")
                return True

            logger.info(f"🔧 Found {len(inconsistent_docs)} inconsistent reports, starting fix...")

            fixed_count = 0
            for doc in inconsistent_docs:
                try:
                    # Add empty reports field for documents missing it
                    update_data = {
                        "$set": {
                            "reports": {},
                            "updated_at": datetime.now()
                        }
                    }

                    result = self.collection.update_one(
                        {"_id": doc["_id"]},
                        update_data
                    )

                    if result.modified_count > 0:
                        fixed_count += 1
                        logger.info(f"✅ Fixed report: {doc.get('analysis_id', 'unknown')}")

                except Exception as e:
                    logger.error(f"❌ Failed to fix report {doc.get('analysis_id', 'unknown')}: {e}")

            logger.info(f"✅ Fix completed, fixed {fixed_count} reports")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to fix inconsistent reports: {e}")
            return False

    def save_report(self, report_data: Dict[str, Any]) -> bool:
        """Save report data (generic method)"""
        if not self.connected:
            logger.warning("MongoDB not connected, skipping save")
            return False

        try:
            # Ensure required fields exist
            if 'analysis_id' not in report_data:
                logger.error("Report data missing analysis_id field")
                return False

            # Add save timestamp
            report_data['saved_at'] = datetime.now()

            # Use upsert operation, update if exists, insert if not
            result = self.collection.replace_one(
                {"analysis_id": report_data['analysis_id']},
                report_data,
                upsert=True
            )

            if result.upserted_id or result.modified_count > 0:
                logger.info(f"✅ Report saved successfully: {report_data['analysis_id']}")
                return True
            else:
                logger.warning(f"⚠️ Report save no changes: {report_data['analysis_id']}")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to save report to MongoDB: {e}")
            return False


# Create global instance
mongodb_report_manager = MongoDBReportManager()
