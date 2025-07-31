import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import chardet
import openpyxl
import PyPDF2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import traceback
import requests
import os
import subprocess
import threading
import time
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import json

from dotenv import load_dotenv

# --- Font Setup for Korean/Unicode ---
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


load_dotenv()
# Prevent repeated logging initialization
if "langsmith_initialized" not in st.session_state:
    logging.langsmith("[Agent] Data_Analysis")
    st.session_state["langsmith_initialized"] = True

# 반복/실패 한계 설정
MAX_AGENT_ITER = 8  # AGENT 최대 반복 허용 횟수
MAX_AGENT_FAIL = 3  # 동일 오류 허용 횟수

# --- Persistent tool-calling status file ---
TOOL_CALLING_STATUS_FILE = "hf_tool_calling_status.json"


def load_tool_calling_status():
    try:
        with open(TOOL_CALLING_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_tool_calling_status(status_dict):
    try:
        with open(TOOL_CALLING_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Failed to save tool-calling status: {e}")


def detect_encoding(file_bytes):
    result = chardet.detect(file_bytes)
    return result["encoding"]


def read_file(file, file_type, encoding=None):
    try:
        if file_type == "csv":
            if encoding is None:
                encoding = detect_encoding(file.getvalue())
            df = pd.read_csv(file, encoding=encoding)
            return df, encoding
        elif file_type == "excel":
            df = pd.read_excel(file)
            return df, "N/A"
        elif file_type == "txt":
            if encoding is None:
                encoding = detect_encoding(file.getvalue())
            content = file.getvalue().decode(encoding)
            delimiter = "\t" if "\t" in content else ","
            try:
                df = pd.read_csv(StringIO(content), sep=delimiter)
                return df, encoding
            except Exception:
                lines = [line.split() for line in content.splitlines()]
                df = pd.DataFrame(lines)
                return df, encoding
        elif file_type == "pdf":
            dfs = []
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lines = [line.split() for line in text.split("\n")]
                    df_page = pd.DataFrame(lines)
                    dfs.append(df_page)
            final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            return final_df, "N/A"
        else:
            return None, None
    except Exception as e:
        st.error(f"파일 파싱 오류: {e}")
        return None, None


def preprocess_dataframe(df):
    # Try to convert all columns to numeric where possible
    for col in df.columns:
        # Remove common numeric formatting (commas, spaces)
        df[col] = df[col].astype(str).str.replace(",", "").str.strip()
        # Try to convert to numeric, if possible
        converted = pd.to_numeric(df[col], errors="coerce")
        # If more than half the values can be converted, use the numeric version
        if converted.notna().sum() >= len(df) / 2:
            df[col] = converted
        else:
            # Otherwise, keep as string and strip whitespace
            df[col] = df[col].astype(str).str.strip()
    # Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)
    # Drop rows where all values are NaN
    df.dropna(how="all", inplace=True)
    return df


def create_agent(
    dataframe, selected_model="gpt-4o", api_base_url=None, api_key=None, temperature=0
):
    # Pass API base URL, API key, and temperature to ChatOpenAI
    return create_pandas_dataframe_agent(
        ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            base_url=api_base_url,
            api_key=api_key,
        ),
        dataframe,
        verbose=True,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        prefix=(
            "You are a professional data analyst and expert in Pandas. "
            "Always check column data types before arithmetic or sorting. "
            "If a column is not numeric, convert it with pd.to_numeric(..., errors='coerce'). "
            "Import all required libraries (pandas, matplotlib, seaborn) at the top of your code. "
            "For visualization:\n"
            "- Always use `plt.figure(figsize=(14,8))`\n"
            "- Use seaborn or matplotlib\n"
            "- Rotate x/y tick labels for readability\n"
            "- Use `st.pyplot(plt.gcf())` instead of plt.show()\n"
            "- Add `plt.close()` after rendering\n"
            "- If there are too many categories, split the plot into multiple figures\n"
            "If you encounter an error, do not repeat the same code more than twice. "
            "If an error persists, explain the cause and suggest a solution in Korean. "
            "The language of final answer should be written in Korean."
        ),
        max_iterations=MAX_AGENT_ITER,
    )


def main():
    st.title("CSV/Excel/TXT/PDF 데이터 LLM 분석 챗봇")

    # 세션 상태 초기화
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None

    # --- Load persistent tool-calling status on startup ---
    if "tool_calling_models" not in st.session_state:
        st.session_state["tool_calling_models"] = load_tool_calling_status()

    with st.sidebar:
        st.header("설정")
        if st.button("대화 초기화"):
            st.session_state["messages"] = []
            st.session_state["df"] = None
            st.session_state["agent"] = None
            st.session_state["uploaded_file"] = None
            st.rerun()
        uploaded_file = st.file_uploader(
            "파일을 업로드 해주세요.",
            type=["csv", "xlsx", "xls", "txt", "pdf"],
            key="file_uploader",
        )
        st.session_state["uploaded_file"] = uploaded_file

        # --- LLM Mode Selection ---
        llm_mode = st.radio(
            "LLM 모드 선택",
            ("API LLM Mode", "Local LLM Mode"),
            key="llm_mode_radio",
        )
        st.session_state.llm_mode = llm_mode

        api_connected = False
        local_connected = False

        if llm_mode == "API LLM Mode":
            st.subheader("LLM API 설정")
            api_base_url = st.text_input(
                "API Base URL",
                value=st.session_state.get("api_base_url", ""),
                placeholder="https://api.openai.com/v1",
                key="api_base_url_input",
            )
            api_key = st.text_input(
                "API Key",
                value=st.session_state.get("api_key", ""),
                type="password",
                placeholder="sk-...",
                key="api_key_input",
            )
            st.session_state.api_base_url = api_base_url
            st.session_state.api_key = api_key

            # --- Dynamic Model List Fetching ---
            def fetch_api_models(api_base_url, api_key):
                url = api_base_url.rstrip("/") + "/models"
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        # OpenAI format: {"data": [{"id": "gpt-4o", ...}, ...]}
                        if "data" in data:
                            return [m["id"] for m in data["data"]]
                        # Azure/other: {"models": ["gpt-4o", ...]}
                        elif "models" in data:
                            return data["models"]
                        # Fallback: try to parse as list
                        elif isinstance(data, list):
                            return data
                        else:
                            return []
                    else:
                        return []
                except Exception:
                    return []

            # Only fetch if base_url or api_key changes
            if (
                "_last_api_base_url" not in st.session_state
                or st.session_state["_last_api_base_url"] != api_base_url
                or st.session_state.get("_last_api_key", None) != api_key
            ):
                st.session_state["_last_api_base_url"] = api_base_url
                st.session_state["_last_api_key"] = api_key
                st.session_state["_api_model_list"] = fetch_api_models(
                    api_base_url, api_key
                )

            api_model_list = st.session_state.get("_api_model_list", [])
            if api_model_list:
                model_options = api_model_list
                model_source_note = "(API에서 자동 감지됨)"
            else:
                model_options = [
                    "gpt-4o",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    "llama2",
                    "mistral",
                    "phi3",
                    "deepseek-coder",
                    "qwen",
                    "mixtral",
                ]
                model_source_note = "(기본 목록)"

            selected_api_model = st.selectbox(
                f"Select a model {model_source_note}",
                model_options,
                index=0 if model_options else None,
                key="api_model_select_dynamic",
            )
            st.session_state.api_model_name = selected_api_model

            # --- Tool-Calling Validation ---
            def validate_tool_calling(api_base_url, api_key, model_name, debug=False):
                url = api_base_url.rstrip("/") + "/chat/completions"
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                # Detect o-series models (gpt-4o, gpt-4o-mini, etc.)
                o_series_prefixes = ["gpt-4o", "gpt-4o-mini", "gpt-4o-"]
                is_o_series = any(
                    model_name.lower().startswith(prefix)
                    for prefix in o_series_prefixes
                )
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "ping_dummy",
                                "description": "Test function.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                            },
                        }
                    ],
                    "tool_choice": "auto",
                }
                # Only add max_tokens for non-o-series models
                if not is_o_series:
                    payload["max_tokens"] = 1
                try:
                    if debug:
                        st.write("[DEBUG] Tool-calling check request:")
                        st.json({"url": url, "headers": headers, "payload": payload})
                    resp = requests.post(url, headers=headers, json=payload, timeout=10)
                    if debug:
                        st.write(f"[DEBUG] Status code: {resp.status_code}")
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.write(resp.text)
                    if resp.status_code == 200:
                        data = resp.json()
                        # Try multiple possible locations for tool_calls
                        tool_calls = None
                        # OpenAI: tool_calls in choices[0].message
                        try:
                            tool_calls = (
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("tool_calls", None)
                            )
                        except Exception:
                            tool_calls = None
                        # Some APIs may put tool_calls at top level or elsewhere
                        if not tool_calls:
                            tool_calls = data.get("tool_calls", None)
                        if not tool_calls:
                            # Try to find tool_calls anywhere in the response
                            import json as _json

                            if "tool_calls" in _json.dumps(data):
                                return (
                                    True,
                                    "⚠️ tool_calls 필드가 응답에 존재하나, 위치가 예상과 다릅니다. (Yellow)",
                                )
                        if tool_calls:
                            return True, "✅ Tool-calling 지원 (Green)"
                        else:
                            return False, "❌ Tool-calling 미지원 (Red)"
                    else:
                        # Try to show error message from response
                        try:
                            err = resp.json()
                        except Exception:
                            err = resp.text
                        return (
                            False,
                            f"❌ Tool-calling 체크 실패: {resp.status_code} - {err}",
                        )
                except Exception as e:
                    return False, f"❌ Tool-calling 체크 오류: {e}"

            # --- Debug toggle in sidebar ---
            debug_tool_calling = st.checkbox(
                "[디버그] Tool-calling 체크 요청/응답 보기",
                value=False,
                key="debug_tool_calling_checkbox",
            )

            # Only validate if model/base_url/api_key changes
            if (
                "_last_validated_model" not in st.session_state
                or st.session_state["_last_validated_model"] != selected_api_model
                or st.session_state["_last_validated_base_url"] != api_base_url
                or st.session_state["_last_validated_api_key"] != api_key
                or st.session_state.get("_last_debug_tool_calling", False)
                != debug_tool_calling
            ):
                st.session_state["_last_validated_model"] = selected_api_model
                st.session_state["_last_validated_base_url"] = api_base_url
                st.session_state["_last_validated_api_key"] = api_key
                st.session_state["_last_debug_tool_calling"] = debug_tool_calling
                valid, msg = validate_tool_calling(
                    api_base_url, api_key, selected_api_model, debug=debug_tool_calling
                )
                st.session_state["_api_model_tool_calling_valid"] = valid
                st.session_state["_api_model_tool_calling_msg"] = msg

            # Show validation result
            valid = st.session_state.get("_api_model_tool_calling_valid", False)
            msg = st.session_state.get("_api_model_tool_calling_msg", "")
            if valid:
                st.success(msg)
            else:
                st.error(msg)

            api_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("api_temperature", 0.0),
                step=0.1,
                key="api_temperature_slider",
            )
            st.session_state.api_temperature = api_temperature

            def check_api_connection():
                url = api_base_url.rstrip("/") + "/models"
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        st.success("✅ LLM API 연결 성공! (models endpoint reachable)")
                        st.session_state.api_connected = True
                        return True
                    else:
                        st.error(
                            f"❌ LLM API 연결 실패: {resp.status_code} - {resp.text}"
                        )
                        st.session_state.api_connected = False
                        return False
                except Exception as e:
                    st.error(f"❌ LLM API 연결 실패: {e}")
                    st.session_state.api_connected = False
                    return False

            if st.button("Check API Status"):
                api_connected = check_api_connection()
            else:
                api_connected = st.session_state.get("api_connected", False)

        elif llm_mode == "Local LLM Mode":
            st.subheader("Local LLM 설정")
            local_platforms = ["Ollama", "Hugging Face"]
            selected_platform = st.selectbox(
                "Local LLM Platform",
                local_platforms,
                key="local_platform_select",
            )
            st.session_state.local_platform = selected_platform

            # --- Dynamic model detection with tool-calling check ---
            local_models = []
            tool_calling_supported = {}
            if selected_platform == "Ollama":
                try:
                    import importlib

                    if importlib.util.find_spec("ollama") is not None:
                        import ollama

                        result = subprocess.run(
                            ["ollama", "list"],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        lines = result.stdout.strip().split("\n")
                        # st.info(f"Ollama 모델 감지됨: {[line.split()[0] for line in lines[1:] if line.strip()]}")
                        for line in lines[1:]:
                            if line.strip():
                                name = line.split()[0]

                                # st.info(f"Ollama 모델 tool-calling 체크: {name}")
                                def check_ollama_tool(model_name):
                                    tools = [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "ping_dummy",
                                                "description": "Test function.",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {},
                                                    "required": [],
                                                },
                                            },
                                        }
                                    ]
                                    try:
                                        # Try with tool_choice first
                                        try:
                                            response = ollama.chat(
                                                model=model_name,
                                                messages=[
                                                    {"role": "user", "content": "ping"}
                                                ],
                                                tools=tools,
                                                tool_choice="auto",
                                            )
                                            # st.info(f"{model_name} tool-call 응답 (with tool_choice): {response}")
                                            if response.get("message", {}).get(
                                                "tool_calls"
                                            ):
                                                return True
                                            else:
                                                return False
                                        except TypeError as e:
                                            if "tool_choice" in str(e):
                                                # Retry without tool_choice
                                                try:
                                                    response = ollama.chat(
                                                        model=model_name,
                                                        messages=[
                                                            {
                                                                "role": "user",
                                                                "content": "ping",
                                                            }
                                                        ],
                                                        tools=tools,
                                                    )
                                                    # st.info(f"{model_name} tool-call 응답 (without tool_choice): {response}")
                                                    if response.get("message", {}).get(
                                                        "tool_calls"
                                                    ):
                                                        return True
                                                    else:
                                                        return False
                                                except Exception as e2:
                                                    # st.warning(f"{model_name} tool-call 체크 중 오류 (no tool_choice): {e2}")
                                                    return False
                                            else:
                                                # st.warning(f"{model_name} tool-call 체크 중 오류 (TypeError): {e}")
                                                return False
                                        except Exception as e:
                                            # st.warning(f"{model_name} tool-call 체크 중 오류 (outer): {e}")
                                            return False
                                    except Exception as e:
                                        # st.warning(f"{model_name} tool-call 체크 중 오류 (global): {e}")
                                        return False

                                supported = check_ollama_tool(name)
                                tool_calling_supported[name] = supported
                                if supported:
                                    local_models.append(name)
                                else:
                                    local_models.append(f"{name} (no tool-calling)")
                        # Add unique key for Ollama selectbox
                        key_suffix = (
                            "hf" if selected_platform == "Hugging Face" else "ollama"
                        )
                        selected_local_model = st.selectbox(
                            "Local LLM Model",
                            local_models,
                            key=f"local_model_select_{key_suffix}",
                        )
                        st.session_state.local_model_name = selected_local_model

                    if selected_local_model == download_option:
                        new_model_id = st.text_input(
                            "Enter Hugging Face model ID (e.g., Qwen/Qwen3-32B):",
                            key="new_hf_model_id_input",
                        )
                        if st.button("Download model", key="download_hf_model_btn"):
                            with st.spinner("New model is downloading now..."):
                                try:
                                    model = AutoModelForCausalLM.from_pretrained(
                                        new_model_id,
                                        device_map="auto",
                                        torch_dtype="auto",
                                        trust_remote_code=True,
                                    )
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        new_model_id,
                                        trust_remote_code=True,
                                    )
                                    st.success(
                                        f"Model {new_model_id} downloaded successfully! Please select it from the list."
                                    )
                                    # Refresh the page to update the model list
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Failed to download model: {e}")
                        st.stop()  # Prevent further UI rendering until download is handled

                    # --- Simple management UI: select model, toggle tool-calling, save ---
                    st.markdown("**Local Hugging Face LLM Tool-Calling Management**")
                    if local_models:
                        selected_model = st.selectbox(
                            "Select a local Hugging Face model to manage tool-calling:",
                            local_models,
                            key="hf_tool_calling_selectbox",
                        )
                        if selected_model:
                            st.info(
                                "Now connecting to Huggingface to get some extra files not cached on this server. Those files are needed to run this llm"
                            )
                        checked = st.checkbox(
                            f"Enable tool-calling for {selected_model}",
                            value=st.session_state["tool_calling_models"].get(
                                selected_model, False
                            ),
                            key=f"tool_calling_checkbox_{selected_model}",
                        )
                        if st.button("Save", key="save_tool_calling_status"):
                            st.session_state["tool_calling_models"][
                                selected_model
                            ] = checked
                            save_tool_calling_status(
                                st.session_state["tool_calling_models"]
                            )
                            st.success(
                                f"Tool-calling status for {selected_model} saved."
                            )
                    else:
                        st.info("No local Hugging Face models found.")
                except Exception as e:
                    st.warning(f"Hugging Face 모델 목록을 불러올 수 없습니다: {e}")
            # Ollama multi-GPU warning
            if selected_platform == "Ollama":
                st.info(
                    "Ollama는 현재 멀티 GPU를 지원하지 않습니다. 한 번에 한 GPU만 사용합니다."
                )

            # Filter for tool-calling supported models, or mark unsupported
            # Only show models that are marked as tool-calling in session state
            # --- Debug prints for model matching ---
            st.write(
                "tool_calling_models:", st.session_state.get("tool_calling_models", {})
            )
            st.write("local_models:", local_models)

            # --- Normalization function for model names ---
            def normalize_model_name(name):
                return name.lower().replace(" ", "")

            # Only show models that are marked as tool-calling in session state (normalized)
            tool_models = []
            normalized_tool_calling = [
                normalize_model_name(k)
                for k, v in st.session_state.get("tool_calling_models", {}).items()
                if v
            ]
            for m in local_models:
                model_id = m.replace(" (tool-calling unknown)", "").replace(
                    " (no tool-calling)", ""
                )
                if normalize_model_name(model_id) in normalized_tool_calling:
                    tool_models.append(m)
            if not tool_models:
                st.warning(
                    "도구 호출(tool-calling)을 지원하는 로컬 모델이 없습니다. 지원되는 모델을 설치하거나 다운로드 해주세요."
                )
                selected_local_model = None
                local_connected = False
            else:
                selected_local_model = st.selectbox(
                    "Local LLM Model",
                    tool_models,
                    key="local_model_select",
                )
                st.session_state.local_model_name = selected_local_model
                if selected_local_model.endswith(
                    "(no tool-calling)"
                ) or selected_local_model.endswith("(tool-calling unknown)"):
                    st.warning(
                        "이 모델은 도구 호출(tool-calling)을 지원하지 않을 수 있습니다. 코드 실행/분석 기능이 제한될 수 있습니다."
                    )

            def check_local_connection():
                platform_endpoints = {
                    "Ollama": "http://localhost:11434/v1/models",
                }
                url = platform_endpoints.get(selected_platform)
                if url is None and selected_platform == "Hugging Face":
                    st.info(
                        "Hugging Face Transformers는 로컬 라이브러리로 동작하므로 별도 연결 확인이 필요 없습니다."
                    )
                    st.session_state.local_connected = True
                    return True
                if url is None:
                    st.error(
                        "이 플랫폼에 대한 연결 확인 방법이 정의되어 있지 않습니다."
                    )
                    st.session_state.local_connected = False
                    return False
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        st.success(
                            f"✅ {selected_platform} 연결 성공! (models endpoint reachable)"
                        )
                        st.session_state.local_connected = True
                        return True
                    else:
                        st.error(
                            f"❌ {selected_platform} 연결 실패: {resp.status_code} - {resp.text}"
                        )
                        st.session_state.local_connected = False
                        return False
                except Exception as e:
                    st.error(f"❌ {selected_platform} 연결 실패: {e}")
                    st.session_state.local_connected = False
                    return False

            if tool_models and st.button("Check Local LLM Connection"):
                local_connected = check_local_connection()
            else:
                local_connected = st.session_state.get("local_connected", False)

        # Enable analysis only after successful connection check
        if (llm_mode == "API LLM Mode" and api_connected) or (
            llm_mode == "Local LLM Mode" and local_connected and tool_models
        ):
            apply_btn = st.button("데이터 분석 시작")
        else:
            st.info("먼저 연결 상태를 확인하세요.")
            apply_btn = False

    # 파일 업로드 없으면 분석 차단
    if st.session_state["uploaded_file"] is None:
        st.session_state["df"] = None
        st.session_state["agent"] = None
        st.warning("파일을 업로드 해주세요.")
        st.stop()

    uploaded_file = st.session_state["uploaded_file"]
    file_name = uploaded_file.name
    ext = file_name.split(".")[-1].lower()
    file_type = (
        "csv"
        if ext == "csv"
        else (
            "excel"
            if ext in ["xls", "xlsx"]
            else "txt" if ext == "txt" else "pdf" if ext == "pdf" else None
        )
    )
    if file_type:
        file_bytes = uploaded_file.getvalue()
        detected_encoding = None
        if file_type in ["csv", "txt"]:
            detected_encoding = detect_encoding(file_bytes)
            st.info(f"자동 감지된 인코딩: {detected_encoding}")
            selected_encoding = st.selectbox(
                "인코딩 선택",
                [detected_encoding, "utf-8", "cp949", "euc-kr"],
                index=0,
            )
        else:
            selected_encoding = None
        file_data, file_encoding = read_file(
            uploaded_file, file_type, encoding=selected_encoding
        )
        if isinstance(file_data, pd.DataFrame):
            df = preprocess_dataframe(file_data)
            st.session_state["df"] = df
            st.subheader("업로드된 데이터 미리보기")
            st.dataframe(df)
        else:
            st.session_state["df"] = None
            st.warning("표 형태의 데이터가 추출되지 않았습니다.")

    if apply_btn and isinstance(st.session_state.get("df"), pd.DataFrame):
        if st.session_state.llm_mode == "API LLM Mode":
            st.session_state["agent"] = create_agent(
                st.session_state["df"],
                st.session_state.api_model_name,
                api_base_url=st.session_state.api_base_url,
                api_key=st.session_state.api_key,
                temperature=st.session_state.api_temperature,
            )
        elif st.session_state.llm_mode == "Local LLM Mode":
            # For local LLM, use default endpoint and selected model
            platform = st.session_state.local_platform
            model_name = st.session_state.local_model_name
            # Set default endpoints for each platform
            platform_endpoints = {
                "Ollama": "http://localhost:11434/v1",
            }
            api_base_url = platform_endpoints.get(platform, None)
            api_key = None
            temperature = 0.0
            if platform == "Hugging Face":
                # For Hugging Face, assume local library call (not API)
                api_base_url = None
            st.session_state["agent"] = create_agent(
                st.session_state["df"],
                model_name,
                api_base_url=api_base_url,
                api_key=api_key,
                temperature=temperature,
            )
        st.success("설정이 완료되었습니다. 대화를 시작해 주세요!")
        st.session_state["messages"] = []

    # --- 대화 출력 (항상 최신 메시지까지 보여줌) ---
    agent = st.session_state.get("agent")
    df = st.session_state.get("df")
    user_input = st.chat_input(
        "궁금한 내용을 입력하세요!", disabled=(df is None or agent is None)
    )

    if user_input:
        st.session_state["messages"].append(("user", user_input))
        # Platform-specific LLM workflow
        if st.session_state.llm_mode == "Local LLM Mode":
            platform = st.session_state.get("local_platform")
            if platform == "Hugging Face":
                # --- Hugging Face Transformers workflow (local inference, no API, no logging) ---
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch

                    model_name = st.session_state.get("local_model_name")
                    if model_name:
                        model_name = model_name.replace(" (tool-calling unknown)", "")
                    hf_token = st.secrets["huggingface"]["token"]
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype="bfloat16",
                        device_map="auto",
                        token=hf_token,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=hf_token,
                    )
                    if hasattr(model, "hf_device_map"):
                        st.info(f"Model device map: {model.hf_device_map}")
                    else:
                        st.info("Model device map info not available.")
                    messages = [{"role": "user", "content": user_input}]
                    tools = []
                    if getattr(tokenizer, "chat_template", None):
                        input_ids = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                    else:
                        prompt = messages[0]["content"]
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    import torch

                    torch.backends.cuda.enable_flash_sdp(True)
                    model.gradient_checkpointing_enable()
                    output = model.generate(
                        input_ids.to(model.device),
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    response = tokenizer.decode(output[0])
                    st.session_state["messages"].append(("assistant", response))
                except Exception as e:
                    st.session_state["messages"].append(
                        ("assistant", f"Hugging Face(로컬) 모델 처리 중 오류: {e}")
                    )
            elif platform == "Ollama":
                # --- Ollama workflow (OpenAI-compatible API or ollama Python package) ---
                if agent is not None and isinstance(df, pd.DataFrame):
                    with st.spinner("LLM이 답변을 생성 중입니다..."):
                        try:
                            fail_count = 0
                            last_error = None
                            for attempt in range(MAX_AGENT_ITER):
                                try:
                                    response = agent.invoke(user_input)
                                    if (
                                        isinstance(response, dict)
                                        and "output" in response
                                    ):
                                        answer = response["output"]
                                    elif isinstance(response, str):
                                        answer = response
                                    else:
                                        answer = "답변을 생성하지 못했습니다."
                                    st.session_state["messages"].append(
                                        ("assistant", answer)
                                    )
                                    if (
                                        isinstance(response, dict)
                                        and "intermediate_steps" in response
                                    ):
                                        for step in response["intermediate_steps"]:
                                            if (
                                                isinstance(step, tuple)
                                                and len(step) >= 2
                                            ):
                                                _, tool_result = step
                                                if (
                                                    hasattr(tool_result, "__class__")
                                                    and tool_result.__class__.__name__
                                                    == "Figure"
                                                ):
                                                    st.pyplot(tool_result)
                                    fig = plt.gcf()
                                    if fig and fig.get_axes():
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    break
                                except Exception as e:
                                    if last_error is not None and str(e) == last_error:
                                        fail_count += 1
                                    else:
                                        fail_count = 1
                                        last_error = str(e)
                                    if fail_count >= MAX_AGENT_FAIL:
                                        st.session_state["messages"].append(
                                            (
                                                "assistant",
                                                f"⚠️ 반복적으로 오류가 발생하여 분석을 중단합니다.\n\n에러 내용: {str(e)}\n\n"
                                                "데이터 파일의 숫자형 컬럼이 문자열로 저장되어 있거나, 입력 데이터에 문제가 있을 수 있습니다. "
                                                "파일을 확인하거나, 데이터 전처리 후 다시 시도해주세요.",
                                            )
                                        )
                                        break
                                    if attempt == MAX_AGENT_ITER - 1:
                                        st.session_state["messages"].append(
                                            (
                                                "assistant",
                                                f"⚠️ 최대 반복 횟수({MAX_AGENT_ITER})에 도달하여 분석을 중단합니다.\n\n"
                                                f"마지막 에러 내용: {str(e)}\n\n"
                                                "데이터 타입/구조 오류, 또는 코드 실행 환경 문제일 수 있습니다. 파일을 확인하거나, 관리자에게 문의하세요.",
                                            )
                                        )
                                        break
                        except Exception as e:
                            st.session_state["messages"].append(
                                (
                                    "assistant",
                                    f"분석 중 치명적 오류 발생: {str(e)}\n\n{traceback.format_exc()}",
                                )
                            )
                else:
                    st.session_state["messages"].append(
                        (
                            "assistant",
                            "분석 가능한 표 데이터가 없습니다. 파일을 먼저 업로드하고 '데이터 분석 시작'을 눌러주세요.",
                        )
                    )
        elif st.session_state.llm_mode == "API LLM Mode":
            try:
                response = agent.invoke({"input": user_input})
                st.session_state["messages"].append(("assistant", response["output"]))
            except Exception as e:
                st.session_state["messages"].append(
                    ("assistant", f"API LLM 처리 중 오류: {e}")
                )

    # Now display all messages, including the new ones
    for i, msg in enumerate(st.session_state["messages"]):
        role, content = msg
        with st.chat_message(role):
            st.markdown(content)


if __name__ == "__main__":
    main()
