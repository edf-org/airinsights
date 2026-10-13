::to run use .\deploy.bat 
::to schedule hourly trigger go to https://console.cloud.google.com/run/jobs/details/us-central1/airinsights-pipeline/triggers?project=edf-aq-data
@echo off

echo Deploying airinsights-pipeline-beta to cloud run jobs

gcloud run jobs deploy airinsights-pipeline-beta ^
  --source . ^
  --project edf-aq-data ^
  --region us-central1 ^
  --tasks 2 ^
  --parallelism 2 ^
  --max-retries 1 ^
  --task-timeout 30m ^
  --memory 4Gi