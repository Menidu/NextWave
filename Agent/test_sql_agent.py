from agents.RM_SQL.workflow import Workflow

def test_sql():
    workflow = Workflow()
    response = workflow.run("How many tier 3 employees are there")
    print(response)
    
def main():
    test_sql()
    
if __name__ == "__main__":
    main()