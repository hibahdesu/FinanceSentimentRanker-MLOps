import subprocess
import sys
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_pipeline():
    """Run the main pipeline (main_pipeline.py)."""
    try:
        logging.info("Running the main pipeline...")
        pipeline_script = os.path.join(os.path.dirname(__file__), 'src', 'pipeline', 'main_pipeline.py')

        # Running the pipeline script via subprocess
        subprocess.run([sys.executable, pipeline_script], check=True)
        logging.info("Pipeline has finished running.")
    except Exception as e:
        logging.error(f"Error running pipeline: {e}")

def run_flask_app():
    """Run the Flask app (app.py) in the background."""
    try:
        logging.info("Starting the Flask app...")
        app_script = os.path.join(os.path.dirname(__file__), 'src', 'api', 'app.py')

        # Running the Flask app as a background process with Popen
        subprocess.Popen([sys.executable, app_script])
        logging.info("Flask app is running in the background.")
    except Exception as e:
        logging.error(f"Error running Flask app: {e}")

def schedule_pipeline_task(scheduler):
    """Schedule the pipeline task."""
    # Add job to scheduler to run the `run_pipeline()` function at a specific time

    # Example of scheduling for daily execution at 2:00 AM:
    scheduler.add_job(run_pipeline, 'cron', day_of_week='mon', hour=2, minute=0)

    # scheduler.add_job(run_pipeline, 'cron', day_of_week='mon', hour=8, minute=0)

    logging.info("Pipeline job has been scheduled.")

def start_scheduler():
    """Start the scheduler."""
    scheduler = BackgroundScheduler()
    scheduler.start()

    # Schedule the pipeline task
    schedule_pipeline_task(scheduler)

    # Keep the main program running to allow the scheduler to work
    try:
        logging.info("Scheduler is running...")
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        # Gracefully shutdown the scheduler
        scheduler.shutdown()
        logging.info("Scheduler has been shut down.")

if __name__ == "__main__":
    # Run the scheduler in the background to run the pipeline at a specific time
    start_scheduler()

    run_flask_app()  
