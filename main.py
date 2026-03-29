from dotenv import load_dotenv
import os 

load_dotenv()

def main():
    print("Hello from langchaincourse!")
    print(os.environ.get('OPENAI_API_KEY'))


if __name__ == "__main__":
    main()
