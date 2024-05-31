import re

class RemoveCommentedText:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "line_comment": ("STRING", {"multiline": False, "default": "#"}),
                "block_comment_start": ("STRING", {"multiline": False, "default": "##"}),
                "block_comment_end": ("STRING", {"multiline": False, "default": "##"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remove_commented_text"
    CATEGORY = "text"
    
    def remove_commented_text(
            self, text, line_comment, block_comment_start, block_comment_end):
        # Remove text enclosed in comment decorators
        text = re.sub(rf'{re.escape(block_comment_start)}[\s\S]+?{re.escape(block_comment_end)}', '', text)
        
        # Then actually remove the decorators
        text = text.replace(f"{re.escape(block_comment_start)}{re.escape(block_comment_end)}", "")
        
        # Remove lines that start with a single "#"
        lines = text.split("\n")
        
        # Skip lines that are commented out, empty, or just a comma
        non_commented_lines = [line for line in lines if not line.strip().startswith(line_comment)]
        
        # Join the non-commented lines back into a single string
        return_text = "\n".join(non_commented_lines)

        return (return_text,)
