#!/usr/bin/env python3
"""
Test script for the rich text logger.
"""

from telemetrics.logger import logger

def test_logger():
    """Test the logger functionality."""
    logger.info("Testing info message")
    logger.debug("Testing debug message")
    logger.warning("Testing warning message")
    logger.error("Testing error message")
    logger.critical("Testing critical message")

    # Test with tags
    logger.info("Testing info with tag", tag="TEST_TAG")
    logger.error("Testing error with tag", tag="ERROR_TAG")

    # Test with message parameter
    logger.info(message="Testing message parameter", tag="PARAM_TEST")

    print("Logger test completed!")

if __name__ == "__main__":
    test_logger()
