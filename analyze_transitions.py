import pandas as pd

def analyze_optimal_transitions():
    try:
        df = pd.read_csv('output/experiment_details.csv')
    except FileNotFoundError:
        print("Error: output/experiment_details.csv not found.")
        return

    # Filter for Optimal strategy
    opt_df = df[df['Algorithm'] == 'Optimal'].sort_values(by=['Service', 'Month'])
    
    if opt_df.empty:
        print("No 'Optimal' strategy data found in logs.")
        return

    print("### Optimal Strategy: Transition Log")
    print("| Month | Service | From | To | Reason/Context |")
    print("|---|---|---|---|---|")

    services = opt_df['Service'].unique()
    
    # Store previous states to detect changes
    # Initialize with 'Greenfield' as per simulation logic
    prev_states = {s: ('Greenfield', 'None', 'None') for s in services}
    
    transitions_found = 0
    
    for month in sorted(opt_df['Month'].unique()):
        month_data = opt_df[opt_df['Month'] == month]
        
        for _, row in month_data.iterrows():
            svc = row['Service']
            curr_state = (row['Mode'], row['Type'], row['Region'])
            
            prev_mode, prev_type, prev_reg = prev_states[svc]
            curr_mode, curr_type, curr_reg = curr_state
            
            # Detect Change
            if curr_state != prev_states[svc]:
                # Format the change
                from_str = f"{prev_mode}"
                if prev_type != "None": from_str += f" ({prev_type})"
                
                to_str = f"**{curr_mode}**"
                if curr_type != "None": to_str += f" ({curr_type})"
                
                # Context guess
                context = ""
                if curr_type == 'Spot': context = "Cost Saving (Reliability Risk)"
                elif curr_mode == 'IaaS': context = "Repatriation"
                elif curr_mode == 'SaaS': context = "Outsourcing"
                
                print(f"| {month} | {svc} | {from_str} | {to_str} | {context} |")
                transitions_found += 1
            
            prev_states[svc] = curr_state

    if transitions_found == 0:
        print("\nNo transitions found! (Strategy stayed static)")

if __name__ == "__main__":
    analyze_optimal_transitions()
