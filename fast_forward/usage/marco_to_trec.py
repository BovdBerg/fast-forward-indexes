from pathlib import Path
import pandas as pd

### Change line below to the path of the run file
run_file_path = '/home/bvdb9/bo-extremely-efficient-query-encoder/outputs/full_eval/dev.rank.tsv'



run_file = Path(run_file_path)
assert run_file.is_file(), f"File {run_file_path} not found."

# Load run_file as dataframe
print(f"Loading run file {run_file}")
df = pd.read_csv(
    run_file,
    sep=r"\s+",
    skipinitialspace=True,
    header=None,
    names=["query_id", "doc_id", "score"],
)
print(f"Completed loading run file {run_file}")

# Transform the dataframe into the format (query_id, iteration, doc_id, rank, score, tag)
print("Transforming the dataframe")
df["iteration"] = "Q0"
df["rank"] = range(1, len(df) + 1)
df["tag"] = "tag"
df = df[["query_id", "iteration", "doc_id", "rank", "score", "tag"]]

# Display the first few rows
print("Head of the transformed dataframe:")
print(df.head())

# Save the transformed dataframe to a new file
output_file_path = run_file.with_name(run_file.stem + ".trec.tsv")
print(f"Saving the transformed dataframe to {output_file_path}")
df.to_csv(
    output_file_path,
    sep="\t",
    columns=["query_id", "iteration", "doc_id", "rank", "score", "tag"],
    index=False,
    header=False,
)
print(f"Completed saving the transformed dataframe to {output_file_path}")
print("Done!")
