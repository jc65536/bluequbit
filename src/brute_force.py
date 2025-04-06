from util import bq_client, load_qasm

client = bq_client()

qc = load_qasm()

res = client.run(qc, job_name="brute_force", shots=1000)

if isinstance(res, list):
    res = res[0]

counts = res.get_counts()

print(max(counts, key=lambda s: counts[s]))
