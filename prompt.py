def generate_user_input(user_message):
    import google.generativeai as genai

    # Securely configure your API key
    GOOGLE_API_KEY = "AIzaSyC_kmbVEli7lAcJ-n8402cJRu5PVGfmTVI"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Define the structured prompt with placeholders for dynamic content
    prompt_template = '''
    Please analyze the user's input and extract the following details to provide a response in the specified format. The response should include the user's budget, intended use, preferred brand, operating system, minimum RAM, storage, and whether a dedicated graphics card is required.

    Use the following constraints for each parameter:

    intended_use: Must be one of the following options: 'General Use', 'Gaming', 'Business', or 'Programming'
    operating_system: Must be one of these six possible options:
    0: 'DOS'
    1: 'Chrome'
    2: 'Mac'
    3: 'Ubuntu'
    4: 'Windows'
    5: 'any' (if not mentioned)
    brand: Choose from one of the following brands:
    0: 'AGB'
    1: 'Acer'
    2: 'Apple'
    3: 'Asus'
    4: 'Avita'
    5: 'Chuwi'
    6: 'Dell'
    7: 'Fujitsu'
    8: 'Gigabyte'
    9: 'HP'
    10: 'Honor'
    11: 'Huawei'
    12: 'Infinix'
    13: 'LG'
    14: 'Lenovo'
    15: 'MSI'
    16: 'Microsoft'
    17: 'Nokia'
    18: 'Realme'
    19: 'Samsung'
    20: 'Ultimus'
    21: 'Xiaomi'
    22: 'Any' (if not mentioned)
    The response should be in the following format:

    user_input += {{
        'budget': <numeric value for budget>,
        'intended_use': '<one of the four possible uses>',
        'brand': '<one of the brands, or 'Any' if not mentioned>',
        'operating_system': '<operating system as per the options>',
        'min_ram': <minimum RAM in GB>,
        'storage': <minimum storage capacity in GB or TB>,
        'graphics_card': <1 if a dedicated graphics card is required, else 0>
    }}
    dont add comments
    also if the budget is mentioned enter the specs acc to the budget
    dont give any in ssd and ram give a number acc to user input ssd have 4 values [256gb,512gb,1tb,2tb] and ram is [8,16,24,32,64,128]
    if budget is not mentioned, automatically suggest one in rupees (concept).

    Input: User Input: "{user_message}"
    '''

    # Embed the user message into the prompt
    prompt = prompt_template.format(user_message=user_message)

    # Call the Gemini API with the generated prompt
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, stream=True)

    # Combine chunks of text
    output_text = ""
    for chunk in response:
        output_text += chunk.text

    # Extract and parse the response into a dictionary
    try:
        local_vars = {}
        exec(output_text, {}, local_vars)
        user_input = local_vars.get('user_input', None)
        if user_input:
            return user_input
        else:
            raise ValueError("Failed to parse user input.")
    except Exception as e:
        print(f"Error parsing the model's output: {e}")
        print("Raw output:", output_text)
        return None
