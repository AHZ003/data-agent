# Deploying DataAgent to Cloud Run

One-time setup is done by hand (it needs your Google account and billing);
after that every push to `main` that passes tests deploys automatically via
`.github/workflows/deploy.yml`.

## What you get

- The Streamlit app on Cloud Run, scaling to zero (`min-instances 0`: no traffic,
  no cost) and capped at 2 instances so a traffic spike can't run up the bill.
- `GOOGLE_API_KEY` in Secret Manager, never in the image or the repo.
- Demo guardrails on (`DEMO_MODE=1`): 10 questions per session on the shared
  key, 10 MB uploads, visitors can paste their own key.
- A $10 budget alert.
- GitHub Actions deploys with **Workload Identity Federation**: GitHub's OIDC
  token is exchanged for short-lived Google credentials, so no long-lived JSON
  service-account key exists anywhere.

## 1. Project and APIs

```bash
export PROJECT_ID=dataagent-demo          # pick a globally unique id
export REGION=us-central1
export REPO=AHZ003/data-agent             # GitHub owner/repo

gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
gcloud billing projects link $PROJECT_ID --billing-account=<BILLING_ACCOUNT_ID>

gcloud services enable run.googleapis.com secretmanager.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com
```

## 2. The API key as a secret

```bash
printf '%s' "<your Gemini API key>" | gcloud secrets create gemini-api-key --data-file=-
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
# Cloud Run's runtime identity (default compute SA) must be able to read it.
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

## 3. First deploy (manual)

```bash
gcloud run deploy dataagent --source . --region $REGION \
  --allow-unauthenticated \
  --max-instances 2 --min-instances 0 --memory 2Gi --cpu 1 --timeout 300 \
  --set-secrets GOOGLE_API_KEY=gemini-api-key:latest \
  --set-env-vars DEMO_MODE=1,DEMO_MAX_QUESTIONS_PER_SESSION=10,DEMO_MAX_UPLOAD_MB=10
```

The command prints the service URL. Open it and ask three sample questions.

## 4. Budget alert

```bash
gcloud billing budgets create --billing-account=<BILLING_ACCOUNT_ID> \
  --display-name="dataagent-demo" --budget-amount=10USD \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

Also set a daily quota cap on the Gemini key in Google AI Studio: the budget
alert only notifies, it does not stop spending.

## 5. GitHub → Google without keys (Workload Identity Federation)

```bash
# A deployer service account for CI.
gcloud iam service-accounts create github-deployer
SA=github-deployer@${PROJECT_ID}.iam.gserviceaccount.com
for role in roles/run.admin roles/cloudbuild.builds.editor roles/artifactregistry.admin \
            roles/storage.admin roles/iam.serviceAccountUser roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA" --role=$role
done

# A pool + provider that trusts GitHub's OIDC tokens for this repo only.
gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc github-oidc \
  --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='${REPO}'"

# Let workflows from this repo impersonate the deployer.
gcloud iam service-accounts add-iam-policy-binding $SA \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${REPO}"

echo "GCP_WIF_PROVIDER=projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/providers/github-oidc"
echo "GCP_DEPLOY_SA=$SA"
```

Add four GitHub repository **variables** (Settings → Secrets and variables →
Actions → Variables): `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER`,
`GCP_DEPLOY_SA`. None of them are secret: without the trust binding above they
grant nothing. The deploy workflow is skipped until they exist.

## Why these choices

| Choice | Reason |
|---|---|
| `min-instances 0` | Idle demo costs nothing; first request after idle takes a few seconds (cold start) |
| `max-instances 2` | Hard ceiling on concurrent spend from a traffic spike |
| Secret Manager | Key rotation without rebuilding; never in image layers or logs |
| WIF instead of a JSON key | Nothing long-lived to leak; access is scoped to this repo |
| Demo limits in the app | Cloud Run caps compute, not LLM spend; the per-session cap and BYO key do |
