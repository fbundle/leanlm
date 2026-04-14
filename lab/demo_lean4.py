from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.utils import remove_marks

# Use an example repository
url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

database = DynamicDatabase()

# Trace the repository to extract theorems and tactics
repo = database.trace_repository(
    url=url,
    commit=commit,
    build_deps=True,
)

if repo is None:
    raise RuntimeError("Failed to trace repository. Ensure you have GITHUB_ACCESS_TOKEN set.")

# Iterate through proven theorems
for theorem in repo.proven_theorems:
    print(f"\n{'='*60}")
    print(f"Theorem: {theorem.full_name}")
    print(f"Statement: {theorem.theorem_statement}")
    print(f"{'='*60}")
    
    if theorem.traced_tactics:
        print("\nProof Steps:")
        for i, tactic_info in enumerate(theorem.traced_tactics):
            # Extract state_before: This is the goal state the model sees as "the problem"
            # remove_marks() strips the <a>...</a> tags used for premise internal tracking
            state_before = remove_marks(tactic_info.state_before).strip()
            tactic = tactic_info.tactic
            
            print(f"\n--- Step {i+1} ---")
            print(f"Goal State (Input to Model):\n{state_before}")
            print(f"Tactic (Model Output): {tactic}")
    else:
        print("\nNo tactics found (possibly a 'by' block that couldn't be traced or a non-tactic proof).")
    
    print(f"\n{'='*60}\n")
