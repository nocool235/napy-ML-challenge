# Team setup: 4 people, 4 × $200 credits, one shared S3 bucket

> **Historical (initial team plan for SageMaker / S3).** The final work ran on one local Windows machine (i7-14700, RTX 4080 Super); see `README.md` and `ARCHITECTURE.md` for the current setup.

## Shape
- **Code** is shared through one private GitHub repo. Each person works on a branch, and merges go through one person.
- **Data and outputs** live in one S3 bucket in the *hub* account, and all four accounts can read and write it.
- **Compute**: each person runs SageMaker in their **own** account, so each person spends their own credits.

This workload is tabular and runs on CPU. There's no deep learning in the baseline. One `ml.m5.2xlarge` or `ml.m5.4xlarge` per person is plenty. **Stop instances when you're not using them.** An idle instance still burns credits.

## 1. Hub account: bucket (≈5 min)
1. Go to S3, then **Create bucket**. Name it something like `aml26-<team>-<random>`. Use one region for everyone, for example `ap-south-1`. Under **Object Ownership**, choose **Bucket owner enforced**. Without this setting, files that teammates upload stay owned by their own accounts.
2. Open the bucket, then **Permissions** → **Bucket policy**. Paste the policy below, with the 12-digit account IDs of the other three teammates:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "TeamRW",
    "Effect": "Allow",
    "Principal": {"AWS": ["arn:aws:iam::<ACCT_B>:root", "arn:aws:iam::<ACCT_C>:root", "arn:aws:iam::<ACCT_D>:root"]},
    "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
    "Resource": ["arn:aws:s3:::<BUCKET>", "arn:aws:s3:::<BUCKET>/*"]
  }]
}
```
3. Upload the dataset once to `s3://<BUCKET>/dataset/` (train/ and test/, exactly as provided).

## 2. Every teammate: SageMaker (≈10 min)
1. In SageMaker AI, choose **Notebooks** → **Create notebook instance**, or use a Studio JupyterLab space. Use `ml.m5.2xlarge` and a 30 GB volume, and let it create a new execution role.
2. In IAM, open that execution role and add an inline policy. Teammates in other accounts need this policy as well as the bucket policy. The hub owner's role usually already has access.
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
    "Resource": ["arn:aws:s3:::<BUCKET>", "arn:aws:s3:::<BUCKET>/*"]
  }]
}
```
3. Open a terminal on the instance and run:
```bash
git clone <your-private-repo> aml && cd aml
aws s3 sync s3://<BUCKET>/dataset/ dataset/
cd code/business_entity_resolution && pip install -r requirements.txt
#   if python < 3.11 on the image:  pip install "pandas<3" scipy scikit-learn lightgbm rapidfuzz   (code works on pandas 2.2 too)
python src/run.py --data-dir ../../dataset --out-dir ../../output --validate-only
```
4. Share results without overwriting each other. Each person writes to their own prefix:
```bash
aws s3 sync ../../output/ s3://<BUCKET>/runs/<yourname>/<short-desc>/ --exclude "debug/test_probs.tsv"
```

## 3. Working rules for the 72 hours
- **One portal account uploads to the leaderboard.** The rules don't allow simultaneous logins, and you only get 5 submissions a day, 15 in total. Before an upload, the person uploading checks that the OOF score in `metrics_train.json` beats the last one.
- Record every upload in `output/submission_log.tsv`, with the commit hash and the LB score. The organisers ask for version history.
- Leave-one-country-out CV (`--cv country`) is your France proxy. Don't accept a change that helps the group CV but hurts the country CV.
- Commit a change only after `--validate-only` shows an OOF gain. Keep a short CHANGELOG line per commit, because it becomes your methodology doc.
- The public LB is only a subset of the test set. Trust OOF over small LB moves.
