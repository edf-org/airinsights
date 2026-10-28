::to run use .\deploy.bat 
::to schedule hourly trigger go to https://console.cloud.google.com/run/jobs/details/us-central1/airinsights-pipeline-beta/triggers?project=edf-aq-data

@echo off

echo Deploying airinsights-pipeline-beta to cloud run jobs

gcloud run jobs deploy airinsights-pipeline-beta ^
  --source . ^
  --project edf-aq-data ^
  --region us-central1 ^
  --tasks 5 ^
  --parallelism 1 ^
  --max-retries 0 ^
  --task-timeout 2h ^
  --memory 16Gi ^
  --cpu 4 ^
  --set-secrets OPENAQ_KEY=openaq:latest,PURPLEAIR_KEY=purpleair:latest