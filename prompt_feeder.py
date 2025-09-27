# prompt_feeder.py

from datetime import datetime

# Define prompts grouped by category
PROMPT_CATEGORIES = {
    "laptops": [
        "best laptops for coding",
        "intel i9 laptops under $1500",
        "lightweight laptops for travel",
        "gaming laptops with RTX 4070",
        "ultrabooks with long battery life"
    ],
    "cameras": [
        "mirrorless camera with 4K and flip screen",
        "dslr for beginners",
        "best cameras for vlogging under $700",
        "action cameras for sports",
        "budget 4K cameras"
    ],
    "tripods": [
        "tripod for wildlife photography",
        "lightweight tripod under $100",
        "heavy-duty tripod for video shooting",
        "travel tripod with ball head",
        "tripods with fluid pan head"
    ],
    "chargers": [
        "best USB-C chargers for iPhone",
        "multi-device charging stations",
        "fast charging car chargers",
        "power banks with fast charging",
        "portable chargers for laptops"
    ],
    "apple": [
        "latest apple watch with fitness tracking",
        "airpods pro with noise cancellation",
        "ipad with stylus support",
        "macbook air m2 13 inch",
        "apple tv 4k streaming device"
    ],
    "audio": [
        "bluetooth speakers under $100",
        "noise cancelling headphones",
        "true wireless earbuds with long battery",
        "soundbars with dolby atmos",
        "budget over-ear headphones"
    ]
}

def get_all_prompts():
    """Return a flat list of all prompts."""
    all_prompts = []
    for prompts in PROMPT_CATEGORIES.values():
        all_prompts.extend(prompts)
    return all_prompts

def get_prompts_by_category():
    """Return the dictionary of prompts categorized."""
    return PROMPT_CATEGORIES

def get_timestamp():
    """Return a current timestamp string suitable for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M")
