def try_wrapper(function, filename, log_path):
    try:
        function()
    except Exception as e:
        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename}: {str(e)}\n")
        print(f"Error {filename}: {str(e)}\n")