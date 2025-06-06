"""
GUI module for ArtRAG System using CustomTkinter.
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
        self.root.grid_columnconfigure(2, weight=1)  # Main content column
        self.root.grid_rowconfigure(0, weight=1)

        # Create sidebar
        self.sidebar = ctk.CTkFrame(self.root, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)  # Empty space
        self.sidebar.grid_propagate(False)  # Maintain fixed width

        # Sidebar title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="ArtRAG", font=ctk.CTkFont(size=26, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(24, 12))

        # Navigation buttons with highlight logic
        self.active_nav_btn = None

        def highlight_nav(btn):
            if self.active_nav_btn and hasattr(self.active_nav_btn, "configure"):
                try:
                    # Reset to default button color
                    self.active_nav_btn.configure(fg_color=("gray75", "gray25"))
                except Exception as e:
                    logger.warning(f"Could not reset navigation button color: {e}")
            btn.configure(fg_color=("#1a1a1a", "#333333"))
            self.active_nav_btn = btn

        self.welcome_btn = ctk.CTkButton(
            self.sidebar,
            text="Search",
            command=lambda: [highlight_nav(self.welcome_btn), self.show_welcome_page()],
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=12,
            hover_color="#2B5CE6",
        )
        self.welcome_btn.grid(row=1, column=0, padx=18, pady=8, sticky="ew")

        self.results_btn = ctk.CTkButton(
            self.sidebar,
            text="Results",
            command=lambda: [highlight_nav(self.results_btn), self.show_results_page()],
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            state="disabled",
            corner_radius=12,
            hover_color="#2B5CE6",
        )
        self.results_btn.grid(row=2, column=0, padx=18, pady=8, sticky="ew")

        self.chat_btn = ctk.CTkButton(
            self.sidebar,
            text="Chat",
            command=lambda: [highlight_nav(self.chat_btn), self.show_chat_page()],
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            state="disabled",
            corner_radius=12,
            hover_color="#2B5CE6",
        )
        self.chat_btn.grid(row=3, column=0, padx=18, pady=8, sticky="ew")

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

        # Vertical separator between sidebar and main content
        self.separator = ctk.CTkFrame(self.root, width=2, fg_color="#222222")
        self.separator.grid(row=0, column=1, sticky="ns")

        # Main content frame with neutral background
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#23272e")
        self.main_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 0))
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
        welcome_container = ctk.CTkFrame(self.main_frame, fg_color="#23272e")
        welcome_container.grid(row=0, column=0, sticky="nsew", padx=32, pady=32)
        welcome_container.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            welcome_container,
            text="ArtRAG - Art Search & Chat System",
            font=ctk.CTkFont(size=36, weight="bold"),
        )
        title_label.grid(row=0, column=0, pady=(48, 24))

        # Subtitle
        subtitle_label = ctk.CTkLabel(
            welcome_container,
            text="Search through art collections and chat with AI about artworks",
            font=ctk.CTkFont(size=18),
        )
        subtitle_label.grid(row=1, column=0, pady=(0, 48))

        # Search section
        search_frame = ctk.CTkFrame(welcome_container, corner_radius=16)
        search_frame.grid(row=2, column=0, sticky="ew", padx=48, pady=24)
        search_frame.grid_columnconfigure(0, weight=1)

        search_label = ctk.CTkLabel(
            search_frame,
            text="Search for artworks:",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        search_label.grid(row=0, column=0, pady=(24, 12))

        # Search input
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Enter your search query (e.g., 'Renaissance paintings', 'Van Gogh', 'still life')",
            height=44,
            font=ctk.CTkFont(size=15),
            corner_radius=10,
        )
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=24, pady=12)
        self.search_entry.bind("<Return>", lambda event: self.perform_search())

        # Search type selection
        search_type_frame = ctk.CTkFrame(search_frame, fg_color="#23272e")
        search_type_frame.grid(row=2, column=0, sticky="ew", padx=24, pady=12)
        search_type_frame.grid_columnconfigure(1, weight=1)

        search_type_label = ctk.CTkLabel(search_type_frame, text="Search Type:")
        search_type_label.grid(row=0, column=0, padx=12, pady=12, sticky="w")

        self.search_type_var = ctk.StringVar(value="advanced_hybrid")
        search_type_menu = ctk.CTkOptionMenu(
            search_type_frame,
            variable=self.search_type_var,
            values=[
                "advanced_hybrid",
                "bm25",
                "rrf_only",
                "semantic",
                "fuzzy",
                "hybrid_scoring",
                "comprehensive",
                "text",
                "metadata",
            ],
        )
        search_type_menu.grid(row=0, column=1, padx=12, pady=12)

        # Search button
        search_button = ctk.CTkButton(
            search_frame,
            text="Search",
            command=self.perform_search,
            height=44,
            font=ctk.CTkFont(size=17, weight="bold"),
            corner_radius=10,
            hover_color="#2B5CE6",
        )
        search_button.grid(row=3, column=0, pady=(12, 24))

        # Status label
        self.status_label = ctk.CTkLabel(
            welcome_container, text="Ready to search...", font=ctk.CTkFont(size=13)
        )
        self.status_label.grid(row=3, column=0, pady=24)

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
        results_container = ctk.CTkScrollableFrame(self.main_frame, fg_color="#23272e")
        results_container.grid(row=0, column=0, sticky="nsew", padx=32, pady=32)
        results_container.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Configure dynamic row weights for proper spacing
        max_rows = (
            len(self.search_results) // 4
        ) + 2  # +2 for title and potential overflow
        for row_idx in range(1, max_rows):
            results_container.grid_rowconfigure(row_idx, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            results_container,
            text=f"Search Results ({len(self.search_results)} found)",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 24))

        # Display results in a grid (4 columns)
        for i, result in enumerate(self.search_results):
            row = (i // 4) + 1
            col = i % 4

            # Create result card with shadow effect
            result_card = ctk.CTkFrame(
                results_container,
                corner_radius=14,
                fg_color="#252a32",
                border_width=2,
                border_color="#2B5CE6",
            )
            result_card.grid(row=row, column=col, padx=14, pady=14, sticky="nsew")

            # Load and display image if available
            if result.image_path and os.path.exists(result.image_path):
                try:
                    image = Image.open(result.image_path)
                    image = image.resize((200, 200), Image.Resampling.LANCZOS)
                    photo = ctk.CTkImage(
                        light_image=image, dark_image=image, size=(200, 200)
                    )

                    image_label = ctk.CTkLabel(
                        result_card, image=photo, text="", corner_radius=10
                    )
                    image_label.grid(row=0, column=0, padx=10, pady=10)
                except Exception as e:
                    logger.error(f"Error loading image {result.image_path}: {e}")
                    placeholder_label = ctk.CTkLabel(
                        result_card,
                        text="\nNo Image",
                        width=200,
                        height=200,
                        font=ctk.CTkFont(size=16),
                    )
                    placeholder_label.grid(row=0, column=0, padx=10, pady=10)
            else:
                placeholder_label = ctk.CTkLabel(
                    result_card,
                    text="\nNo Image",
                    width=200,
                    height=200,
                    font=ctk.CTkFont(size=16),
                )
                placeholder_label.grid(row=0, column=0, padx=10, pady=10)

            # Artwork info
            info_frame = ctk.CTkFrame(result_card, fg_color="#23272e")
            info_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

            title_text = (
                result.title[:30] + "..." if len(result.title) > 30 else result.title
            )
            title_label = ctk.CTkLabel(
                info_frame,
                text=title_text,
                font=ctk.CTkFont(size=15, weight="bold"),
                wraplength=180,
            )
            title_label.grid(row=0, column=0, padx=5, pady=5)

            author_text = (
                result.author[:25] + "..." if len(result.author) > 25 else result.author
            )
            author_label = ctk.CTkLabel(
                info_frame,
                text=f"by {author_text}",
                font=ctk.CTkFont(size=13),
                wraplength=180,
            )
            author_label.grid(row=1, column=0, padx=5, pady=(0, 5))

            # Score
            score_label = ctk.CTkLabel(
                info_frame,
                text=f"Score: {result.relevance_score:.3f}",
                font=ctk.CTkFont(size=11),
            )
            score_label.grid(row=2, column=0, padx=5, pady=(0, 5))

            # Chat button
            chat_button = ctk.CTkButton(
                result_card,
                text="Chat",
                command=lambda r=result: self.start_chat_with_artwork(r),
                height=34,
                font=ctk.CTkFont(size=13),
                corner_radius=8,
                hover_color="#2B5CE6",
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
        chat_container = ctk.CTkFrame(self.main_frame, fg_color="#23272e")
        chat_container.grid(row=0, column=0, sticky="nsew", padx=32, pady=32)
        chat_container.grid_columnconfigure(1, weight=1)
        chat_container.grid_rowconfigure(1, weight=1)

        # Left panel - Artwork info
        artwork_panel = ctk.CTkFrame(chat_container, width=300, fg_color="#252a32")
        artwork_panel.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 14))
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

                image_label = ctk.CTkLabel(
                    artwork_panel, image=photo, text="", corner_radius=10
                )
                image_label.grid(row=0, column=0, padx=10, pady=10)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                placeholder_label = ctk.CTkLabel(
                    artwork_panel,
                    text="\nNo Image Available",
                    width=280,
                    height=280,
                    font=ctk.CTkFont(size=16),
                )
                placeholder_label.grid(row=0, column=0, padx=10, pady=10)
        else:
            placeholder_label = ctk.CTkLabel(
                artwork_panel,
                text="\nNo Image Available",
                width=280,
                height=280,
                font=ctk.CTkFont(size=16),
            )
            placeholder_label.grid(row=0, column=0, padx=10, pady=10)

        # Artwork details
        details_frame = ctk.CTkScrollableFrame(
            artwork_panel, height=200, fg_color="#23272e"
        )
        details_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        details_text = f"""Title: {self.current_artwork.title}\n\nArtist: {self.current_artwork.author}\n\nContent: {self.current_artwork.content}\n\nRelevance Score: {self.current_artwork.relevance_score:.3f}\n\nSearch Type: {self.current_artwork.search_type}"""

        details_label = ctk.CTkLabel(
            details_frame,
            text=details_text,
            font=ctk.CTkFont(size=13),
            justify="left",
            wraplength=260,
        )
        details_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Right panel - Chat
        chat_title = ctk.CTkLabel(
            chat_container,
            text=f"Chat about: {self.current_artwork.title[:40]}...",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        chat_title.grid(row=0, column=1, pady=(0, 14), sticky="ew")

        # Chat history
        self.chat_display = ctk.CTkScrollableFrame(chat_container, fg_color="#23272e")
        self.chat_display.grid(row=1, column=1, sticky="nsew", pady=(0, 14))
        self.chat_display.grid_columnconfigure(0, weight=1)

        # Chat input frame
        input_frame = ctk.CTkFrame(chat_container, fg_color="#252a32")
        input_frame.grid(row=2, column=1, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.chat_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Ask me anything about this artwork...",
            font=ctk.CTkFont(size=15),
            corner_radius=10,
        )
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(12, 6), pady=12)
        self.chat_entry.bind("<Return>", lambda event: self.send_message())

        send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            font=ctk.CTkFont(size=15),
            corner_radius=10,
            hover_color="#2B5CE6",
        )
        send_button.grid(row=0, column=1, padx=(6, 12), pady=12)

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

        # Add placeholder for assistant response
        self.chat_history.append({"role": "assistant", "content": "Thinking..."})
        assistant_message_index = len(self.chat_history) - 1
        self.update_chat_display()

        # Generate streaming response in a separate thread
        def generate_streaming_response():
            try:
                full_response = ""

                # Use the streaming generator
                for chunk in self.ollama_rag.generate_response_stream(
                    query=message,
                    context_results=[self.current_artwork],
                    max_context_length=2000,
                    temperature=0.3,
                ):
                    full_response += chunk

                    # Update the assistant message with the accumulated response
                    # Update every chunk for real-time streaming effect
                    self.chat_history[assistant_message_index][
                        "content"
                    ] = full_response
                    self.root.after(0, self.update_chat_display)

                # Ensure we have a final response
                if not full_response.strip():
                    self.chat_history[assistant_message_index][
                        "content"
                    ] = "Sorry, I couldn't generate a response."
                    self.root.after(0, self.update_chat_display)

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                self.chat_history[assistant_message_index]["content"] = error_msg
                self.root.after(0, self.update_chat_display)

        threading.Thread(target=generate_streaming_response, daemon=True).start()

    def update_chat_display(self):
        """Update the chat display with current history."""
        # Clear current display
        for widget in self.chat_display.winfo_children():
            widget.destroy()

        # Display chat history with alternating backgrounds
        for i, message in enumerate(self.chat_history):
            bg_color = "#23272e" if i % 2 == 0 else "#252a32"
            message_frame = ctk.CTkFrame(self.chat_display, fg_color=bg_color)
            message_frame.grid(row=i, column=0, sticky="ew", padx=10, pady=5)
            message_frame.grid_columnconfigure(0, weight=1)

            # Role indicator
            role_color = "#2B5CE6" if message["role"] == "user" else "#00B4D8"
            role_text = "You" if message["role"] == "user" else "AI Assistant"

            role_label = ctk.CTkLabel(
                message_frame,
                text=role_text,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=role_color,
            )
            role_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

            # Message content
            content_label = ctk.CTkLabel(
                message_frame,
                text=message["content"],
                font=ctk.CTkFont(size=13),
                wraplength=600,
                justify="left",
            )
            content_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Scroll to bottom with error handling
        try:
            self.root.after(
                100, lambda: self.chat_display._parent_canvas.yview_moveto(1.0)
            )
        except Exception as e:
            logger.warning(f"Could not scroll chat display: {e}")

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
