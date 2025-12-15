"""Command-line interface for running the Copilot proxy."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

import uvicorn

from .config import (
    get_api_key, set_api_key, get_base_url, set_base_url, get_config_file,
    is_first_run, ensure_complete_config,
    set_context_length, get_context_length,
    set_model_name, get_model_name,
    set_temperature, get_temperature
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434


def interactive_setup(is_first: bool = True) -> None:
    """Run interactive setup for users."""
    if is_first:
        print("🚀 Welcome to Copilot Proxy!")
        print("This appears to be your first time running the proxy.")
        print("Let's set up your configuration.\n")
    else:
        print("⚙️  Copilot Proxy Setup")
        print("Let's configure your settings.\n")

    # Get API key
    while True:
        api_key = input("Please enter your Z.AI API key (or press Enter to skip): ").strip()
        if not api_key:
            print("⚠️  No API key provided. You can set it later with:")
            print("   copilot-proxy config set-api-key <your-key>")
            break

        if len(api_key) < 10:
            print("❌ API key seems too short. Please check your API key.")
            continue

        set_api_key(api_key)
        print("✅ API key saved successfully!")
        break

    # Get optional base URL
    base_url = input("\nEnter custom base URL (optional, press Enter to use default): ").strip()
    if base_url:
        set_base_url(base_url)
        print("✅ Custom base URL saved!")

    # Ensure all default values are saved to config
    ensure_complete_config()

    # Show config location
    config_path = get_config_file()
    print(f"\n📁 Configuration saved to: {config_path}")
    print("\nYou're all set! You can now start the proxy with:")
    print("   copilot-proxy serve")
    print("\nYou can always change your configuration later using:")
    print("   copilot-proxy config --help")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Copilot GLM proxy server.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command (default behavior)
    serve_parser = subparsers.add_parser("serve", help="Start the proxy server")
    serve_parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host interface to bind (default: {DEFAULT_HOST}).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind (default: {DEFAULT_PORT}).",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (useful for development).",
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        help="Log level passed to Uvicorn (default: info).",
    )
    serve_parser.add_argument(
        "--proxy-app",
        default="copilot_proxy.app:app",
        help=(
            "Dotted path to the FastAPI application passed to Uvicorn "
            "(default: copilot_proxy.app:app)."
        ),
    )

    # Config commands
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Config actions")

    # Set API key
    set_key_parser = config_subparsers.add_parser("set-api-key", help="Set API key in config")
    set_key_parser.add_argument("api_key", help="Your Z.AI API key")

    # Get API key
    config_subparsers.add_parser("get-api-key", help="Show current API key from config")

    # Set base URL
    set_url_parser = config_subparsers.add_parser("set-base-url", help="Set base URL in config")
    set_url_parser.add_argument("base_url", help="Custom base URL for API")

    # Get base URL
    config_subparsers.add_parser("get-base-url", help="Show current base URL from config")

    # Set context length
    set_ctx_parser = config_subparsers.add_parser("set-context-length", help="Set context length in config")
    set_ctx_parser.add_argument("context_length", type=int, help="Context length (e.g. 64000, 128000)")

    # Get context length
    config_subparsers.add_parser("get-context-length", help="Show current context length from config")

    # Set model name
    set_model_parser = config_subparsers.add_parser("set-model", help="Set default model name in config")
    set_model_parser.add_argument("model_name", help="Model name (e.g. GLM-4.6)")

    # Get model name
    config_subparsers.add_parser("get-model", help="Show current default model name from config")

    # Set temperature
    set_temp_parser = config_subparsers.add_parser("set-temperature", help="Set temperature override in config")
    set_temp_parser.add_argument("temperature", type=float, help="Temperature override (e.g. 0.1, 0.7)")

    # Get temperature
    config_subparsers.add_parser("get-temperature", help="Show current temperature override from config")

    # Show config path
    config_subparsers.add_parser("show-path", help="Show where the config file is stored")

    # Interactive setup
    config_subparsers.add_parser("setup", help="Run interactive setup wizard")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Check for first run and show interactive setup
    # Only run interactive setup when no command is provided (default behavior)
    if is_first_run() and args.command is None:
        interactive_setup()
        return

    # Handle config commands
    if args.command == "config":
        if args.config_action == "set-api-key":
            set_api_key(args.api_key)
            print("API key saved to config file.")
        elif args.config_action == "get-api-key":
            api_key = get_api_key()
            if api_key:
                print(f"Current API key: {api_key}")
            else:
                print("No API key found in config file.")
        elif args.config_action == "set-base-url":
            set_base_url(args.base_url)
            print("Base URL saved to config file.")
        elif args.config_action == "get-base-url":
            base_url = get_base_url()
            if base_url:
                print(f"Current base URL: {base_url}")
            else:
                print("No base URL found in config file.")
        elif args.config_action == "set-context-length":
            set_context_length(args.context_length)
            print("Context length saved to config file.")
        elif args.config_action == "get-context-length":
            print(f"Current context length: {get_context_length()}")
        elif args.config_action == "set-model":
            set_model_name(args.model_name)
            print("Model name saved to config file.")
        elif args.config_action == "get-model":
            model_name = get_model_name()
            if model_name:
                print(f"Current model name: {model_name}")
            else:
                print("No model name set in config (using default).")
        elif args.config_action == "set-temperature":
            set_temperature(args.temperature)
            print("Temperature saved to config file.")
        elif args.config_action == "get-temperature":
            temp = get_temperature()
            if temp is not None:
                print(f"Current temperature: {temp}")
            else:
                print("No temperature override set in config.")
        elif args.config_action == "show-path":
            config_path = get_config_file()
            print(f"Config file location: {config_path}")
            print(f"Config directory: {config_path.parent}")
        elif args.config_action == "setup":
            interactive_setup(is_first=False)
        else:
            parser.print_help()
        return

    # Default to serve command if no command specified
    if args.command is None or args.command == "serve":
        # For serve command, use default values if not provided
        host = getattr(args, 'host', DEFAULT_HOST)
        port = getattr(args, 'port', DEFAULT_PORT)
        reload = getattr(args, 'reload', False)
        log_level = getattr(args, 'log_level', 'info')
        proxy_app = getattr(args, 'proxy_app', 'copilot_proxy.app:app')

        uvicorn.run(
            proxy_app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
