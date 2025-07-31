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
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import sqlite3
from contextlib import contextmanager

# Enhanced imports for better table handling
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    logging.langsmith("[Agent] Enhanced_Data_Analysis")
    st.session_state["langsmith_initialized"] = True

# Enhanced configuration
MAX_AGENT_ITER = 8
MAX_AGENT_FAIL = 3
TOOL_CALLING_STATUS_FILE = "hf_tool_calling_status.json"

# --- Enhanced Data Structures for Table Analysis ---

class QueryType(Enum):
    """Types of queries that can be handled"""
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    STATISTICAL = "statistical"
    VISUALIZATION = "visualization"
    GENERAL = "general"

@dataclass
class TableSchema:
    """Enhanced table schema information"""
    columns: List[str]
    data_types: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    primary_key: Optional[str] = None
    foreign_keys: List[str] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = []
        if self.constraints is None:
            self.constraints = {}

@dataclass
class QueryIntent:
    """Parsed query intent"""
    query_type: QueryType
    target_columns: List[str]
    filters: Dict[str, Any]
    aggregations: List[str]
    sort_by: Optional[str] = None
    sort_order: str = "asc"
    limit: Optional[int] = None
    group_by: Optional[str] = None

class TableAnalyzer:
    """Enhanced table analyzer for structured data processing"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.schema = self._analyze_schema()
        self._create_indexes()
    
    def _analyze_schema(self) -> TableSchema:
        """Analyze table structure and create enhanced schema"""
        columns = list(self.df.columns)
        data_types = {}
        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for col in columns:
            dtype = str(self.df[col].dtype)
            data_types[col] = dtype
            
            # Enhanced type detection
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_columns.append(col)
            elif self.df[col].nunique() < len(self.df) * 0.1:  # Less than 10% unique values
                categorical_columns.append(col)
        
        return TableSchema(
            columns=columns,
            data_types=data_types,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns
        )
    
    def _create_indexes(self):
        """Create optimized indexes for fast retrieval"""
        # Create SQLite in-memory database for fast queries
        self.conn = sqlite3.connect(':memory:')
        self.df.to_sql('data', self.conn, index=False, if_exists='replace')
        
        # Create indexes on numeric columns for fast range queries
        for col in self.schema.numeric_columns:
            try:
                self.conn.execute(f'CREATE INDEX idx_{col} ON data({col})')
            except:
                pass
    
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a column"""
        if column not in self.df.columns:
            return {}
        
        stats = {
            'name': column,
            'dtype': str(self.df[column].dtype),
            'count': len(self.df[column]),
            'null_count': self.df[column].isnull().sum(),
            'unique_count': self.df[column].nunique()
        }
        
        if column in self.schema.numeric_columns:
            stats.update({
                'min': float(self.df[column].min()),
                'max': float(self.df[column].max()),
                'mean': float(self.df[column].mean()),
                'median': float(self.df[column].median()),
                'std': float(self.df[column].std()),
                'q25': float(self.df[column].quantile(0.25)),
                'q75': float(self.df[column].quantile(0.75))
            })
        elif column in self.schema.categorical_columns:
            value_counts = self.df[column].value_counts()
            stats.update({
                'top_values': value_counts.head(5).to_dict(),
                'value_distribution': (value_counts / len(self.df)).to_dict()
            })
        
        return stats
    
    def execute_structured_query(self, query_intent: QueryIntent) -> pd.DataFrame:
        """Execute structured query using SQL for better performance"""
        sql_query = self._build_sql_query(query_intent)
        
        try:
            result = pd.read_sql_query(sql_query, self.conn)
            return result
        except Exception as e:
            st.warning(f"SQL query failed, falling back to pandas: {e}")
            return self._execute_pandas_query(query_intent)
    
    def _build_sql_query(self, query_intent: QueryIntent) -> str:
        """Build SQL query from query intent"""
        select_clause = ", ".join(query_intent.target_columns) if query_intent.target_columns else "*"
        
        # Handle aggregations
        if query_intent.aggregations:
            agg_clauses = []
            for agg in query_intent.aggregations:
                if agg.lower() in ['sum', 'avg', 'count', 'min', 'max']:
                    agg_clauses.append(f"{agg.upper()}({query_intent.target_columns[0]})")
            if agg_clauses:
                select_clause = ", ".join(agg_clauses)
        
        sql = f"SELECT {select_clause} FROM data"
        
        # Add WHERE clause for filters
        if query_intent.filters:
            where_conditions = []
            for col, value in query_intent.filters.items():
                if isinstance(value, (list, tuple)):
                    placeholders = ','.join(['?' for _ in value])
                    where_conditions.append(f"{col} IN ({placeholders})")
                else:
                    where_conditions.append(f"{col} = ?")
            if where_conditions:
                sql += f" WHERE {' AND '.join(where_conditions)}"
        
        # Add GROUP BY
        if query_intent.group_by:
            sql += f" GROUP BY {query_intent.group_by}"
        
        # Add ORDER BY
        if query_intent.sort_by:
            sql += f" ORDER BY {query_intent.sort_by} {query_intent.sort_order.upper()}"
        
        # Add LIMIT
        if query_intent.limit:
            sql += f" LIMIT {query_intent.limit}"
        
        return sql
    
    def _execute_pandas_query(self, query_intent: QueryIntent) -> pd.DataFrame:
        """Fallback to pandas operations"""
        result = self.df.copy()
        
        # Apply filters
        for col, value in query_intent.filters.items():
            if isinstance(value, (list, tuple)):
                result = result[result[col].isin(value)]
            else:
                result = result[result[col] == value]
        
        # Apply aggregations
        if query_intent.aggregations:
            agg_dict = {}
            for agg in query_intent.aggregations:
                if agg.lower() in ['sum', 'avg', 'count', 'min', 'max']:
                    agg_dict[query_intent.target_columns[0]] = agg.lower()
            if agg_dict:
                result = result.groupby(query_intent.group_by).agg(agg_dict).reset_index()
        
        return result

class QueryParser:
    """Enhanced query parser for understanding user intent"""
    
    def __init__(self, table_analyzer: TableAnalyzer):
        self.analyzer = table_analyzer
        self.schema = table_analyzer.schema
    
    def parse_query(self, query: str) -> QueryIntent:
        """Parse natural language query into structured intent"""
        query_lower = query.lower()
        
        # Initialize query intent
        intent = QueryIntent(
            query_type=QueryType.GENERAL,
            target_columns=[],
            filters={},
            aggregations=[]
        )
        
        # Detect query type
        if any(word in query_lower for word in ['sum', 'total', '합계', '총합']):
            intent.query_type = QueryType.AGGREGATION
            intent.aggregations.append('sum')
        elif any(word in query_lower for word in ['average', 'mean', '평균']):
            intent.query_type = QueryType.AGGREGATION
            intent.aggregations.append('avg')
        elif any(word in query_lower for word in ['count', '개수', '수량']):
            intent.query_type = QueryType.AGGREGATION
            intent.aggregations.append('count')
        elif any(word in query_lower for word in ['filter', 'where', '조건', '필터']):
            intent.query_type = QueryType.FILTERING
        elif any(word in query_lower for word in ['compare', '비교']):
            intent.query_type = QueryType.COMPARISON
        elif any(word in query_lower for word in ['trend', 'trends', '추세']):
            intent.query_type = QueryType.TREND_ANALYSIS
        elif any(word in query_lower for word in ['statistics', 'stats', '통계']):
            intent.query_type = QueryType.STATISTICAL
        elif any(word in query_lower for word in ['plot', 'chart', 'graph', '시각화', '그래프']):
            intent.query_type = QueryType.VISUALIZATION
        
        # Extract column references
        for col in self.schema.columns:
            if col.lower() in query_lower or col.replace('_', ' ').lower() in query_lower:
                intent.target_columns.append(col)
        
        # Extract numeric filters
        numeric_patterns = [
            r'(\w+)\s*(?:>|>=|<=|<|=)\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:>|>=|<=|<|=)\s*(\w+)',
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if len(match) == 2:
                    col, value = match
                    # Try to find matching column
                    for schema_col in self.schema.columns:
                        if col.lower() in schema_col.lower():
                            try:
                                numeric_value = float(value.replace(',', ''))
                                intent.filters[schema_col] = numeric_value
                                break
                            except ValueError:
                                pass
        
        return intent

class EnhancedDataAgent:
    """Enhanced data agent with hybrid symbolic + LLM approach"""
    
    def __init__(self, df: pd.DataFrame, llm, temperature: float = 0):
        self.df = df
        self.llm = llm
        self.temperature = temperature
        self.table_analyzer = TableAnalyzer(df)
        self.query_parser = QueryParser(self.table_analyzer)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create specialized tools
        self.tools = self._create_tools()
        
        # Create enhanced agent
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[BaseTool]:
        """Create specialized tools for table operations"""
        from langchain.tools import tool
        
        @tool
        def get_table_schema() -> str:
            """Get comprehensive table schema information"""
            schema = self.table_analyzer.schema
            return json.dumps({
                'columns': schema.columns,
                'data_types': schema.data_types,
                'numeric_columns': schema.numeric_columns,
                'categorical_columns': schema.categorical_columns,
                'datetime_columns': schema.datetime_columns,
                'row_count': len(self.df),
                'column_count': len(schema.columns)
            }, indent=2, ensure_ascii=False)
        
        @tool
        def get_column_statistics(column_name: str) -> str:
            """Get detailed statistics for a specific column"""
            stats = self.table_analyzer.get_column_stats(column_name)
            return json.dumps(stats, indent=2, ensure_ascii=False)
        
        @tool
        def execute_structured_query(query_description: str) -> str:
            """Execute a structured query based on natural language description"""
            try:
                # Parse the query
                intent = self.query_parser.parse_query(query_description)
                
                # Execute the query
                result = self.table_analyzer.execute_structured_query(intent)
                
                # Format the result
                if len(result) > 100:
                    return f"Query returned {len(result)} rows. First 100 rows:\n{result.head(100).to_string()}"
                else:
                    return result.to_string()
            except Exception as e:
                return f"Query execution failed: {str(e)}"
        
        @tool
        def create_visualization(chart_type: str, x_column: str, y_column: str, title: str = "") -> str:
            """Create a visualization chart"""
            try:
                plt.figure(figsize=(14, 8))
                
                if chart_type.lower() in ['bar', 'barplot']:
                    plt.bar(self.df[x_column], self.df[y_column])
                elif chart_type.lower() in ['line', 'lineplot']:
                    plt.plot(self.df[x_column], self.df[y_column])
                elif chart_type.lower() in ['scatter', 'scatterplot']:
                    plt.scatter(self.df[x_column], self.df[y_column])
                elif chart_type.lower() in ['histogram', 'hist']:
                    plt.hist(self.df[x_column], bins=30)
                elif chart_type.lower() in ['box', 'boxplot']:
                    plt.boxplot(self.df[x_column])
                else:
                    return f"Unsupported chart type: {chart_type}"
                
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                if title:
                    plt.title(title)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(plt.gcf())
                plt.close()
                
                return f"Created {chart_type} chart for {x_column} vs {y_column}"
            except Exception as e:
                return f"Visualization failed: {str(e)}"
        
        @tool
        def perform_statistical_analysis(columns: str = "") -> str:
            """Perform comprehensive statistical analysis"""
            try:
                if columns:
                    target_cols = [col.strip() for col in columns.split(',')]
                    numeric_cols = [col for col in target_cols if col in self.table_analyzer.schema.numeric_columns]
                else:
                    numeric_cols = self.table_analyzer.schema.numeric_columns
                
                if not numeric_cols:
                    return "No numeric columns found for statistical analysis"
                
                analysis = {}
                for col in numeric_cols:
                    analysis[col] = self.table_analyzer.get_column_stats(col)
                
                return json.dumps(analysis, indent=2, ensure_ascii=False)
            except Exception as e:
                return f"Statistical analysis failed: {str(e)}"
        
        return [get_table_schema, get_column_statistics, execute_structured_query, 
                create_visualization, perform_statistical_analysis]
    
    def _create_agent(self) -> AgentExecutor:
        """Create enhanced agent with structured tools"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data analyst specializing in table data analysis. 
            You have access to specialized tools for handling structured data efficiently.
            
            Key capabilities:
            1. Schema analysis and understanding
            2. Structured query execution (SQL-based for performance)
            3. Statistical analysis
            4. Data visualization
            5. Column-specific operations
            
            Always:
            - Use the appropriate tool for the task
            - Provide context about the data structure when relevant
            - Explain your analysis steps in Korean
            - Handle numeric data with precision
            - Consider data types when performing operations
            
            For numeric queries, prefer structured queries over text-based reasoning.
            For visualizations, use the create_visualization tool with appropriate parameters.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            max_iterations=MAX_AGENT_ITER,
            handle_parsing_errors=True
        )
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Invoke the enhanced agent"""
        try:
            # First, try structured approach for numeric queries
            if self._is_numeric_query(query):
                intent = self.query_parser.parse_query(query)
                if intent.query_type in [QueryType.AGGREGATION, QueryType.FILTERING, QueryType.STATISTICAL]:
                    result = self.table_analyzer.execute_structured_query(intent)
                    return {
                        "output": f"구조화된 쿼리 결과:\n{result.to_string()}",
                        "method": "structured",
                        "result": result
                    }
            
            # Fall back to LLM agent for complex queries
            response = self.agent.invoke({"input": query})
            return {
                "output": response["output"],
                "method": "llm_agent",
                "result": response
            }
        except Exception as e:
            return {
                "output": f"오류 발생: {str(e)}",
                "method": "error",
                "error": str(e)
            }
    
    def _is_numeric_query(self, query: str) -> bool:
        """Check if query is primarily numeric/computational"""
        numeric_keywords = [
            'sum', 'total', 'average', 'mean', 'count', 'min', 'max',
            '합계', '총합', '평균', '개수', '최소', '최대',
            'filter', 'where', '조건', '필터',
            'statistics', 'stats', '통계'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in numeric_keywords)

# --- Enhanced File Processing Functions ---

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
    """Enhanced preprocessing with better type detection"""
    # Try to convert all columns to numeric where possible
    for col in df.columns:
        # Remove common numeric formatting (commas, spaces, currency symbols)
        df[col] = df[col].astype(str).str.replace(",", "").str.replace("$", "").str.replace("₩", "").str.strip()
        
        # Try to convert to numeric, if possible
        converted = pd.to_numeric(df[col], errors="coerce")
        
        # If more than half the values can be converted, use the numeric version
        if converted.notna().sum() >= len(df) / 2:
            df[col] = converted
        else:
            # Try datetime conversion
            try:
                datetime_converted = pd.to_datetime(df[col], errors="coerce")
                if datetime_converted.notna().sum() >= len(df) / 2:
                    df[col] = datetime_converted
                else:
                    # Otherwise, keep as string and strip whitespace
                    df[col] = df[col].astype(str).str.strip()
            except:
                # Otherwise, keep as string and strip whitespace
                df[col] = df[col].astype(str).str.strip()
    
    # Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)
    
    # Drop rows where all values are NaN
    df.dropna(how="all", inplace=True)
    
    return df

def create_enhanced_agent(
    dataframe, selected_model="gpt-4o", api_base_url=None, api_key=None, temperature=0
):
    """Create enhanced agent with hybrid approach"""
    llm = ChatOpenAI(
        model=selected_model,
        temperature=temperature,
        base_url=api_base_url,
        api_key=api_key,
    )
    
    return EnhancedDataAgent(dataframe, llm, temperature)

# --- Main Application ---

def main():
    st.title("향상된 CSV/Excel/TXT/PDF 데이터 LLM 분석 챗봇")
    st.markdown("**하이브리드 접근법: 구조화된 쿼리 + LLM 에이전트**")

    # 세션 상태 초기화
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None
    if "table_analyzer" not in st.session_state:
        st.session_state["table_analyzer"] = None

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
            st.session_state["table_analyzer"] = None
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

            # Model selection (simplified for brevity)
            model_options = [
                "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "llama2", "mistral", "phi3"
            ]
            selected_api_model = st.selectbox(
                "Select a model",
                model_options,
                key="api_model_select",
            )
            st.session_state.api_model_name = selected_api_model

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
                        st.success("✅ LLM API 연결 성공!")
                        st.session_state.api_connected = True
                        return True
                    else:
                        st.error(f"❌ LLM API 연결 실패: {resp.status_code}")
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
            # Simplified local LLM setup for brevity
            st.info("Local LLM 설정은 기존 코드와 동일하게 구현됩니다.")
            local_connected = True

        # Enable analysis only after successful connection check
        if (llm_mode == "API LLM Mode" and api_connected) or (
            llm_mode == "Local LLM Mode" and local_connected
        ):
            apply_btn = st.button("향상된 데이터 분석 시작")
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
            
            # Create table analyzer
            if st.session_state["table_analyzer"] is None:
                st.session_state["table_analyzer"] = TableAnalyzer(df)
            
            st.subheader("업로드된 데이터 미리보기")
            st.dataframe(df)
            
            # Show enhanced schema information
            with st.expander("📊 향상된 테이블 스키마 정보"):
                analyzer = st.session_state["table_analyzer"]
                schema = analyzer.schema
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 행 수", len(df))
                    st.metric("총 열 수", len(schema.columns))
                with col2:
                    st.metric("숫자형 열", len(schema.numeric_columns))
                    st.metric("범주형 열", len(schema.categorical_columns))
                with col3:
                    st.metric("날짜형 열", len(schema.datetime_columns))
                
                st.write("**숫자형 열:**", ", ".join(schema.numeric_columns) if schema.numeric_columns else "없음")
                st.write("**범주형 열:**", ", ".join(schema.categorical_columns) if schema.categorical_columns else "없음")
                st.write("**날짜형 열:**", ", ".join(schema.datetime_columns) if schema.datetime_columns else "없음")
        else:
            st.session_state["df"] = None
            st.warning("표 형태의 데이터가 추출되지 않았습니다.")

    if apply_btn and isinstance(st.session_state.get("df"), pd.DataFrame):
        if st.session_state.llm_mode == "API LLM Mode":
            st.session_state["agent"] = create_enhanced_agent(
                st.session_state["df"],
                st.session_state.api_model_name,
                api_base_url=st.session_state.api_base_url,
                api_key=st.session_state.api_key,
                temperature=st.session_state.api_temperature,
            )
        elif st.session_state.llm_mode == "Local LLM Mode":
            # Simplified local LLM setup
            st.session_state["agent"] = create_enhanced_agent(
                st.session_state["df"],
                "local-model",
                temperature=0.0,
            )
        
        st.success("향상된 설정이 완료되었습니다. 대화를 시작해 주세요!")
        st.session_state["messages"] = []

    # --- 대화 출력 ---
    agent = st.session_state.get("agent")
    df = st.session_state.get("df")
    user_input = st.chat_input(
        "궁금한 내용을 입력하세요!", disabled=(df is None or agent is None)
    )

    if user_input:
        st.session_state["messages"].append(("user", user_input))
        
        if agent is not None and isinstance(df, pd.DataFrame):
            with st.spinner("향상된 분석 엔진이 답변을 생성 중입니다..."):
                try:
                    response = agent.invoke(user_input)
                    
                    if response["method"] == "structured":
                        st.info("🔧 구조화된 쿼리 엔진 사용")
                    elif response["method"] == "llm_agent":
                        st.info("🤖 LLM 에이전트 사용")
                    
                    st.session_state["messages"].append(("assistant", response["output"]))
                    
                except Exception as e:
                    st.session_state["messages"].append(
                        ("assistant", f"분석 중 오류 발생: {str(e)}")
                    )
        else:
            st.session_state["messages"].append(
                ("assistant", "분석 가능한 표 데이터가 없습니다. 파일을 먼저 업로드하고 '향상된 데이터 분석 시작'을 눌러주세요.")
            )

    # Display all messages
    for i, msg in enumerate(st.session_state["messages"]):
        role, content = msg
        with st.chat_message(role):
            st.markdown(content)

def load_tool_calling_status():
    try:
        with open(TOOL_CALLING_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

if __name__ == "__main__":
    main() 