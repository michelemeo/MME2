import json

def save_dict_json(path, data_dict):
    with open(path, 'w') as f:
        json.dump(data_dict, f, indent=4)


def update_dict_value(path, key, new_value):
    # Load dict from JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Update the value
    data[key] = new_value

    # Save back to file
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    return data