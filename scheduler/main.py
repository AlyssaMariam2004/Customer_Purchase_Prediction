import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from trainer import maybe_retrain_model
from data_ingestion import sync_data
from logger import setup_logging

setup_logging()
scheduler = BlockingScheduler()

now = datetime.now()
scheduler.add_job(sync_data, 'interval', minutes=2, next_run_time=now)
scheduler.add_job(maybe_retrain_model, 'interval', minutes=2, next_run_time=now + timedelta(seconds=5))

if __name__ == "__main__":
    logging.info("Starting scheduler service...")
    sync_data()
    maybe_retrain_model()
    scheduler.start()
