import torch
from jinja2 import Template
import json
import os
import re

class TherapeuticChatHandler:
    def __init__(self, model_loader):
        """
        Initialize therapeutic chat handler with loaded model
        Specialized for mental health and counseling conversations
        
        Args:
            model_loader: Instance of GemmaModelLoader
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.adapter_path = model_loader.adapter_path
        self.conversation_history = []
        self.chat_template = self._load_chat_template()
        
        # Therapeutic conversation parameters
        self.system_prompt = """You are a compassionate and experienced mental health counselor. 
Your role is to provide supportive, empathetic, and professional guidance to individuals 
seeking help with their mental health concerns. Listen carefully, validate their feelings, 
and offer constructive advice based on therapeutic principles. Always maintain a warm, 
non-judgmental, and supportive tone."""
        
    def _load_chat_template(self):
        """Load the Jinja2 chat template for therapeutic conversations"""
        try:
            template_path = os.path.join(self.adapter_path, "chat_template.jinja")
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_content = f.read()
                print(f"✓ Chat template loaded: {template_path}")
                return Template(template_content)
            else:
                print(f"⚠ Chat template not found at: {template_path}, using default")
                return None
        except Exception as e:
            print(f"⚠ Could not load chat template: {str(e)}")
            return None
    
    def format_conversation(self, messages, include_system=True):
        """
        Format conversation using the chat template
        Includes system prompt for therapeutic context
        """
        # Add system prompt if not already present
        if include_system and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        if self.chat_template:
            try:
                return self.chat_template.render(messages=messages)
            except Exception as e:
                print(f"⚠ Template rendering error: {str(e)}")
        
        # Fallback formatting for Gemma format
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"<start_of_turn>system\n{content}<end_of_turn>\n"
            elif role == "user":
                formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant" or role == "model":
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        
        # Add the prompt for model response
        formatted += "<start_of_turn>model\n"
        return formatted
    
    def add_to_history(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def preprocess_user_message(self, message):
        """
        Preprocess user message for therapeutic context
        Maintains empathetic understanding
        """
        # Clean up the message
        message = message.strip()
        
        # Return as-is for therapeutic conversations
        # The model is fine-tuned to understand mental health context
        return message
    
    def postprocess_response(self, response):
        """
        Postprocess model response for therapeutic appropriateness
        """
        # Clean up response
        response = response.strip()
        
        # Remove any incomplete sentences at the end
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > 1 and not re.search(r'[.!?]$', sentences[-1]):
            response = ' '.join(sentences[:-1])
        
        return response
    
    def generate_response(self, user_message, max_length=512, temperature=0.7, 
                         top_p=0.9, top_k=50, use_history=True, 
                         repetition_penalty=1.1):
        """
        Generate therapeutic response from the model
        
        Args:
            user_message: User input text (their concern/question)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0.7 for balanced responses)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            use_history: Whether to use conversation history
            repetition_penalty: Penalty for repetitive text
            
        Returns:
            Generated therapeutic response text
        """
        try:
            # Preprocess user message
            processed_message = self.preprocess_user_message(user_message)
            
            # Add user message to history
            self.add_to_history("user", processed_message)
            
            # Prepare messages for formatting
            if use_history:
                messages = self.conversation_history.copy()
            else:
                messages = [{"role": "user", "content": processed_message}]
            
            # Format conversation with system prompt
            prompt = self.format_conversation(messages, include_system=True)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract only the model's response
            response = self._extract_model_response(full_response, prompt)
            
            # Postprocess for therapeutic appropriateness
            response = self.postprocess_response(response)
            
            # Add to history
            self.add_to_history("model", response)
            
            return response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error. Please try again. Error: {str(e)}"
            print(f"✗ Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    def _extract_model_response(self, full_response, prompt):
        """Extract only the model's response from full output"""
        try:
            # Remove the input prompt
            response = full_response[len(prompt):] if len(full_response) > len(prompt) else full_response
            
            # Remove end-of-turn tokens and clean up
            response = response.replace("<end_of_turn>", "").strip()
            response = response.replace("<start_of_turn>", "").strip()
            
            # Remove any remaining special tokens
            response = re.sub(r'<[^>]+>', '', response).strip()
            
            # Split by turn markers and get first response
            if "user" in response.lower() or "model" in response.lower():
                parts = re.split(r'\b(user|model)\b', response, flags=re.IGNORECASE)
                if len(parts) > 0:
                    response = parts[0].strip()
            
            return response
        except Exception as e:
            print(f"⚠ Error extracting response: {str(e)}")
            return full_response
    
    def get_history(self):
        """Return conversation history"""
        return self.conversation_history
    
    def set_history(self, history):
        """Set conversation history from external source"""
        self.conversation_history = history
        print(f"✓ History updated with {len(history)} messages")
    
    def get_conversation_summary(self):
        """Get a summary of the current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for msg in self.conversation_history if msg["role"] == "user"),
            "model_messages": sum(1 for msg in self.conversation_history if msg["role"] == "model")
        }
