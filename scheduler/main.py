from apscheduler.schedulers.blocking import BlockingScheduler
from scheduler.data_ingestion import sync_data
from scheduler.trainer import maybe_retrain_model
from app.logger import setup_logging

setup_logging()

scheduler = BlockingScheduler()
scheduler.add_job(sync_data, 'interval', minutes=10)
scheduler.add_job(maybe_retrain_model, 'interval', minutes=10)

if __name__ == "__main__":
    scheduler.start()
