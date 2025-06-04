"""
GUI module for ArtRAG System using CustomTkinter.
Provides a sleek interface for searching and chatting about artworks.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import os
from pathlib import Path
import threading
from PIL import Image
import logging

from retrieval_gui import ThreadSafeArtSearch
from ollama_gui import OllamaArtRAGGUI, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CustomTkinter
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class ArtRAGModernGUI:
    """Modern GUI application for the ArtRAG System using CustomTkinter."""

    def __init__(
        self,
        db_path: str = "art_database.db",
        chroma_path: str = "./chroma_db",
        image_base_path: str = "./data/img",
    ):
        """Initialize the modern GUI application."""
        # Initialize backend systems
        self.art_search = ThreadSafeArtSearch(db_path=db_path, chroma_path=chroma_path)
        self.ollama_rag = OllamaArtRAGGUI(
            db_path=db_path,
            chroma_path=chroma_path,
            image_base_path=image_base_path,
        )
        self.image_base_path = Path(image_base_path)

        # GUI state
        self.current_artwork = None
        self.search_results = []
        self.chat_history = []

        # Create main window
        self.root = ctk.CTk()
        self.root.title("ArtRAG - Art Search & Chat System")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        # Create main container with sidebar
        self.setup_main_layout()

        # Start with welcome page
        self.show_welcome_page()

    def setup_main_layout(self):
        """Setup the main layout with sidebar navigation."""
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Create sidebar
        self.sidebar = ctk.CTkFrame(self.root, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)  # Empty space

        # Sidebar title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="ArtRAG", font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Navigation buttons
        self.welcome_btn = ctk.CTkButton(
            self.sidebar,
            text="🏠 Search",
            command=self.show_welcome_page,
            height=40,
            font=ctk.CTkFont(size=14),
        )
        self.welcome_btn.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.results_btn = ctk.CTkButton(
            self.sidebar,
            text="📋 Results",
            command=self.show_results_page,
            height=40,
            font=ctk.CTkFont(size=14),
            state="disabled",
        )
        self.results_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.chat_btn = ctk.CTkButton(
            self.sidebar,
            text="💬 Chat",
            command=self.show_chat_page,
            height=40,
            font=ctk.CTkFont(size=14),
            state="disabled",
        )
        self.chat_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Appearance mode switch
        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        # Main content frame
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        """Change the appearance mode."""
        ctk.set_appearance_mode(new_appearance_mode)

    def clear_main_frame(self):
        """Clear the main frame for new content."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_welcome_page(self):
        """Display the welcome/search page."""
        self.clear_main_frame()

        # Welcome container
        welcome_container = ctk.CTkFrame(self.main_frame)
        welcome_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        welcome_container.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            welcome_container,
            text="ArtRAG - Art Search & Chat System",
            font=ctk.CTkFont(size=32, weight="bold"),
        )
        title_label.grid(row=0, column=0, pady=(40, 20))

        # Subtitle
        subtitle_label = ctk.CTkLabel(
            welcome_container,
            text="Search through art collections and chat with AI about artworks",
            font=ctk.CTkFont(size=16),
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 40))

        # Search section
        search_frame = ctk.CTkFrame(welcome_container)
        search_frame.grid(row=2, column=0, sticky="ew", padx=40, pady=20)
        search_frame.grid_columnconfigure(0, weight=1)

        search_label = ctk.CTkLabel(
            search_frame,
            text="Search for artworks:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        search_label.grid(row=0, column=0, pady=(20, 10))

        # Search input
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Enter your search query (e.g., 'Renaissance paintings', 'Van Gogh', 'still life')",
            height=40,
            font=ctk.CTkFont(size=14),
        )
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        self.search_entry.bind("<Return>", lambda event: self.perform_search())

        # Search type selection
        search_type_frame = ctk.CTkFrame(search_frame)
        search_type_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

        search_type_label = ctk.CTkLabel(search_type_frame, text="Search Type:")
        search_type_label.grid(row=0, column=0, padx=10, pady=10)

        self.search_type_var = ctk.StringVar(value="comprehensive")
        search_type_menu = ctk.CTkOptionMenu(
            search_type_frame,
            variable=self.search_type_var,
            values=["comprehensive", "semantic", "text", "metadata"],
        )
        search_type_menu.grid(row=0, column=1, padx=10, pady=10)

        # Search button
        search_button = ctk.CTkButton(
            search_frame,
            text="🔍 Search",
            command=self.perform_search,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        search_button.grid(row=3, column=0, pady=(10, 20))

        # Status label
        self.status_label = ctk.CTkLabel(
            welcome_container, text="Ready to search...", font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=3, column=0, pady=20)

    def perform_search(self):
        """Perform the search operation."""
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query.")
            return

        # Update status
        self.status_label.configure(text="Searching...")
        self.root.update()

        # Perform search in a separate thread
        def search_thread():
            try:
                search_type = self.search_type_var.get()
                self.search_results = self.ollama_rag.search_artworks(
                    query=query, k=12, search_type=search_type, include_images=True
                )

                # Update UI in main thread
                self.root.after(0, self.search_completed, len(self.search_results))

            except Exception as e:
                logger.error(f"Search error: {e}")
                self.root.after(
                    0,
                    lambda e=e: self.status_label.configure(
                        text=f"Search error: {str(e)}"
                    ),
                )

        threading.Thread(target=search_thread, daemon=True).start()

    def search_completed(self, num_results):
        """Handle search completion."""
        self.status_label.configure(text=f"Found {num_results} results")
        if num_results > 0:
            self.results_btn.configure(state="normal")
            self.show_results_page()
        else:
            messagebox.showinfo("No Results", "No artworks found for your query.")

    def show_results_page(self):
        """Display the search results page."""
        if not self.search_results:
            messagebox.showinfo("No Results", "Please perform a search first.")
            return

        self.clear_main_frame()

        # Results container with scrollable frame
        results_container = ctk.CTkScrollableFrame(self.main_frame)
        results_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        results_container.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Title
        title_label = ctk.CTkLabel(
            results_container,
            text=f"Search Results ({len(self.search_results)} found)",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))

        # Display results in a grid (4 columns)
        for i, result in enumerate(self.search_results):
            row = (i // 4) + 1
            col = i % 4

            # Create result card
            result_card = ctk.CTkFrame(results_container)
            result_card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            # Load and display image if available
            if result.image_path and os.path.exists(result.image_path):
                try:
                    image = Image.open(result.image_path)
                    image = image.resize((200, 200), Image.Resampling.LANCZOS)
                    photo = ctk.CTkImage(
                        light_image=image, dark_image=image, size=(200, 200)
                    )

                    image_label = ctk.CTkLabel(result_card, image=photo, text="")
                    image_label.grid(row=0, column=0, padx=10, pady=10)
                except Exception as e:
                    logger.error(f"Error loading image {result.image_path}: {e}")
                    placeholder_label = ctk.CTkLabel(
                        result_card,
                        text="🖼️\nNo Image",
                        width=200,
                        height=200,
                        font=ctk.CTkFont(size=16),
                    )
                    placeholder_label.grid(row=0, column=0, padx=10, pady=10)
            else:
                placeholder_label = ctk.CTkLabel(
                    result_card,
                    text="🖼️\nNo Image",
                    width=200,
                    height=200,
                    font=ctk.CTkFont(size=16),
                )
                placeholder_label.grid(row=0, column=0, padx=10, pady=10)

            # Artwork info
            info_frame = ctk.CTkFrame(result_card)
            info_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

            title_text = (
                result.title[:30] + "..." if len(result.title) > 30 else result.title
            )
            title_label = ctk.CTkLabel(
                info_frame,
                text=title_text,
                font=ctk.CTkFont(size=14, weight="bold"),
                wraplength=180,
            )
            title_label.grid(row=0, column=0, padx=5, pady=5)

            author_text = (
                result.author[:25] + "..." if len(result.author) > 25 else result.author
            )
            author_label = ctk.CTkLabel(
                info_frame,
                text=f"by {author_text}",
                font=ctk.CTkFont(size=12),
                wraplength=180,
            )
            author_label.grid(row=1, column=0, padx=5, pady=(0, 5))

            # Score
            score_label = ctk.CTkLabel(
                info_frame,
                text=f"Score: {result.relevance_score:.3f}",
                font=ctk.CTkFont(size=10),
            )
            score_label.grid(row=2, column=0, padx=5, pady=(0, 5))

            # Chat button
            chat_button = ctk.CTkButton(
                result_card,
                text="💬 Chat",
                command=lambda r=result: self.start_chat_with_artwork(r),
                height=30,
                font=ctk.CTkFont(size=12),
            )
            chat_button.grid(row=2, column=0, padx=10, pady=(0, 10))

    def start_chat_with_artwork(self, artwork: SearchResult):
        """Start chat with a specific artwork."""
        self.current_artwork = artwork
        self.chat_history = []
        self.chat_btn.configure(state="normal")
        self.show_chat_page()

    def show_chat_page(self):
        """Display the chat page."""
        if not self.current_artwork:
            messagebox.showinfo(
                "No Artwork", "Please select an artwork from search results first."
            )
            return

        self.clear_main_frame()

        # Chat container
        chat_container = ctk.CTkFrame(self.main_frame)
        chat_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        chat_container.grid_columnconfigure(1, weight=1)
        chat_container.grid_rowconfigure(1, weight=1)

        # Left panel - Artwork info
        artwork_panel = ctk.CTkFrame(chat_container, width=300)
        artwork_panel.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 10))
        artwork_panel.grid_propagate(False)

        # Artwork image
        if self.current_artwork.image_path and os.path.exists(
            self.current_artwork.image_path
        ):
            try:
                image = Image.open(self.current_artwork.image_path)
                image = image.resize((280, 280), Image.Resampling.LANCZOS)
                photo = ctk.CTkImage(
                    light_image=image, dark_image=image, size=(280, 280)
                )

                image_label = ctk.CTkLabel(artwork_panel, image=photo, text="")
                image_label.grid(row=0, column=0, padx=10, pady=10)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                placeholder_label = ctk.CTkLabel(
                    artwork_panel,
                    text="🖼️\nNo Image Available",
                    width=280,
                    height=280,
                    font=ctk.CTkFont(size=16),
                )
                placeholder_label.grid(row=0, column=0, padx=10, pady=10)
        else:
            placeholder_label = ctk.CTkLabel(
                artwork_panel,
                text="🖼️\nNo Image Available",
                width=280,
                height=280,
                font=ctk.CTkFont(size=16),
            )
            placeholder_label.grid(row=0, column=0, padx=10, pady=10)

        # Artwork details
        details_frame = ctk.CTkScrollableFrame(artwork_panel, height=200)
        details_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        details_text = f"""Title: {self.current_artwork.title}

Artist: {self.current_artwork.author}

Content: {self.current_artwork.content}

Relevance Score: {self.current_artwork.relevance_score:.3f}

Search Type: {self.current_artwork.search_type}"""

        details_label = ctk.CTkLabel(
            details_frame,
            text=details_text,
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=260,
        )
        details_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Right panel - Chat
        chat_title = ctk.CTkLabel(
            chat_container,
            text=f"Chat about: {self.current_artwork.title[:40]}...",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        chat_title.grid(row=0, column=1, pady=(0, 10), sticky="ew")

        # Chat history
        self.chat_display = ctk.CTkScrollableFrame(chat_container)
        self.chat_display.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        self.chat_display.grid_columnconfigure(0, weight=1)

        # Chat input frame
        input_frame = ctk.CTkFrame(chat_container)
        input_frame.grid(row=2, column=1, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.chat_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Ask me anything about this artwork...",
            font=ctk.CTkFont(size=14),
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=10)
        self.chat_entry.bind("<Return>", lambda event: self.send_message())

        send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            font=ctk.CTkFont(size=14),
        )
        send_button.grid(row=0, column=1, padx=(5, 10), pady=10)

        # Update chat display
        self.update_chat_display()

    def send_message(self):
        """Send a chat message."""
        message = self.chat_entry.get().strip()
        if not message:
            return

        # Clear input
        self.chat_entry.delete(0, tk.END)

        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        self.update_chat_display()

        # Generate response in a separate thread
        def generate_response():
            try:
                response = self.ollama_rag.generate_response(
                    query=message,
                    context_results=[self.current_artwork],
                    max_context_length=2000,
                    temperature=0.7,
                )

                # Add response to history
                self.chat_history.append({"role": "assistant", "content": response})
                self.root.after(0, self.update_chat_display)

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                self.chat_history.append({"role": "assistant", "content": error_msg})
                self.root.after(0, self.update_chat_display)

        threading.Thread(target=generate_response, daemon=True).start()

    def update_chat_display(self):
        """Update the chat display with current history."""
        # Clear current display
        for widget in self.chat_display.winfo_children():
            widget.destroy()

        # Display chat history
        for i, message in enumerate(self.chat_history):
            message_frame = ctk.CTkFrame(self.chat_display)
            message_frame.grid(row=i, column=0, sticky="ew", padx=10, pady=5)
            message_frame.grid_columnconfigure(0, weight=1)

            # Role indicator
            role_color = "#2B5CE6" if message["role"] == "user" else "#00B4D8"
            role_text = "You" if message["role"] == "user" else "AI Assistant"

            role_label = ctk.CTkLabel(
                message_frame,
                text=role_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=role_color,
            )
            role_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

            # Message content
            content_label = ctk.CTkLabel(
                message_frame,
                text=message["content"],
                font=ctk.CTkFont(size=12),
                wraplength=600,
                justify="left",
            )
            content_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Scroll to bottom
        self.chat_display._parent_canvas.yview_moveto(1.0)

    def run(self):
        """Start the GUI application."""
        logger.info("Starting ArtRAG Modern GUI...")
        self.root.mainloop()

    def close(self):
        """Close the application and cleanup resources."""
        try:
            if hasattr(self.ollama_rag, "close"):
                self.ollama_rag.close()
            if hasattr(self.art_search, "close"):
                self.art_search.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.root.quit()


def main():
    """Main entry point for the application."""
    try:
        # Initialize the GUI with the correct paths
        app = ArtRAGModernGUI(
            db_path="src/art_database.db",
            chroma_path="src/chroma_db",
            image_base_path="data/img",
        )
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {str(e)}")
    finally:
        if "app" in locals():
            app.close()


if __name__ == "__main__":
    main()
