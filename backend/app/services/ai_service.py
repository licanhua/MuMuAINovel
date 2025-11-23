"""AI服务封装 - 统一的多提供商接口"""
from typing import Optional, AsyncGenerator, List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import httpx
import hashlib
from app.config import settings as app_settings
from app.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Provider Enums
# ============================================================================

class AIProvider(str, Enum):
    """AI提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    CUSTOM = "custom"  # 自定义OpenAI兼容API


# ============================================================================
# HTTP Client Pool Management
# ============================================================================

_http_client_pool: Dict[str, httpx.AsyncClient] = {}


def _get_client_key(provider: str, base_url: Optional[str], api_key: str) -> str:
    """生成HTTP客户端的唯一键"""
    key_hash = hashlib.md5(api_key.encode()).hexdigest()[:8]
    url_part = base_url or "default"
    return f"{provider}_{url_part}_{key_hash}"


def _get_or_create_http_client(
    provider: str,
    base_url: Optional[str],
    api_key: str
) -> httpx.AsyncClient:
    """获取或创建HTTP客户端（复用连接）"""
    global _http_client_pool
    
    client_key = _get_client_key(provider, base_url, api_key)
    
    if client_key in _http_client_pool:
        client = _http_client_pool[client_key]
        if not client.is_closed:
            logger.debug(f"♻️ 复用HTTP客户端: {client_key}")
            return client
        else:
            logger.warning(f"⚠️ HTTP客户端已关闭，重新创建: {client_key}")
            del _http_client_pool[client_key]
    
    limits = httpx.Limits(
        max_keepalive_connections=50,
        max_connections=100,
        keepalive_expiry=30.0
    )
    
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=60.0,
            read=180.0,
            write=60.0,
            pool=60.0
        ),
        limits=limits,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    )
    
    _http_client_pool[client_key] = client
    logger.info(f"✅ 创建新HTTP客户端并加入池: {client_key} (池大小: {len(_http_client_pool)})")
    
    return client


async def cleanup_http_clients():
    """清理所有HTTP客户端（应用关闭时调用）"""
    global _http_client_pool
    
    logger.info(f"🧹 开始清理HTTP客户端池 (共 {len(_http_client_pool)} 个客户端)")
    
    for key, client in list(_http_client_pool.items()):
        try:
            if not client.is_closed:
                await client.aclose()
                logger.debug(f"✅ 关闭HTTP客户端: {key}")
        except Exception as e:
            logger.error(f"❌ 关闭HTTP客户端失败 {key}: {e}")
    
    _http_client_pool.clear()
    logger.info("✅ HTTP客户端池清理完成")


# ============================================================================
# Provider Interface
# ============================================================================

class AIProviderInterface(ABC):
    """AI提供商统一接口"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.http_client = http_client
    
    @abstractmethod
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成文本（支持工具调用）
        
        Returns:
            {
                "content": "生成的文本",
                "tool_calls": [...],  # 如果有工具调用
                "finish_reason": "stop"
            }
        """
        pass
    
    @abstractmethod
    async def generate_text_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        pass


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(AIProviderInterface):
    """OpenAI提供商实现"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        super().__init__(api_key, base_url, http_client)
        from openai import AsyncOpenAI
        
        if not http_client:
            http_client = _get_or_create_http_client("openai", base_url, api_key)
        
        client_kwargs = {
            "api_key": api_key,
            "http_client": http_client
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncOpenAI(**client_kwargs)
        logger.info("✅ OpenAI提供商初始化成功")
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用OpenAI生成文本"""
        try:
            logger.info(f"🔵 调用OpenAI API - 模型: {model}")
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # 添加工具参数
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    if tool_choice == "required":
                        kwargs["tool_choice"] = "required"
                    elif tool_choice == "auto":
                        kwargs["tool_choice"] = "auto"
                    elif tool_choice == "none":
                        kwargs["tool_choice"] = "none"
            
            response = await self.client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            message = choice.message
            
            # 检查工具调用
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return {
                "content": message.content or "",
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": choice.finish_reason
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI API调用失败: {str(e)}")
            raise
    
    async def generate_text_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """使用OpenAI流式生成文本"""
        try:
            logger.info(f"🔵 调用OpenAI流式API - 模型: {model}")
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
            
        except Exception as e:
            logger.error(f"❌ OpenAI流式API调用失败: {str(e)}")
            raise


# ============================================================================
# Anthropic Provider
# ============================================================================

class AnthropicProvider(AIProviderInterface):
    """Anthropic提供商实现"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        super().__init__(api_key, base_url, http_client)
        from anthropic import AsyncAnthropic
        
        if not http_client:
            http_client = _get_or_create_http_client("anthropic", base_url, api_key)
        
        client_kwargs = {
            "api_key": api_key,
            "http_client": http_client
        }
        
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncAnthropic(**client_kwargs)
        logger.info("✅ Anthropic提供商初始化成功")
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用Anthropic生成文本"""
        try:
            logger.info(f"🔵 调用Anthropic API - 模型: {model}")
            
            # 提取system消息
            system_prompt = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_messages.append(msg)
            
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            # 添加工具参数
            if tools:
                kwargs["tools"] = tools
                if tool_choice == "required":
                    kwargs["tool_choice"] = {"type": "any"}
                elif tool_choice == "auto":
                    kwargs["tool_choice"] = {"type": "auto"}
            
            response = await self.client.messages.create(**kwargs)
            
            # 处理响应
            tool_calls = []
            content_text = ""
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input
                        }
                    })
                elif block.type == "text":
                    content_text += block.text
            
            return {
                "content": content_text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": response.stop_reason
            }
            
        except Exception as e:
            logger.error(f"❌ Anthropic API调用失败: {str(e)}")
            raise
    
    async def generate_text_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """使用Anthropic流式生成文本"""
        try:
            logger.info(f"🔵 调用Anthropic流式API - 模型: {model}")
            
            # 提取system消息
            system_prompt = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_messages.append(msg)
            
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
            
        except Exception as e:
            logger.error(f"❌ Anthropic流式API调用失败: {str(e)}")
            raise


# ============================================================================
# Gemini Provider
# ============================================================================

class GeminiProvider(AIProviderInterface):
    """Google Gemini提供商实现（使用官方google-generativeai库）"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ):
        super().__init__(api_key, base_url, http_client)
        
        try:
            import google.generativeai as genai
            
            # 配置API密钥
            genai.configure(api_key=api_key)
            
            self.genai = genai
            logger.info("✅ Gemini提供商初始化成功")
        except ImportError:
            logger.error("❌ 未安装google-generativeai库，请运行: pip install google-generativeai")
            raise ImportError("请安装google-generativeai: pip install google-generativeai")
    
    def _convert_messages_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """转换消息格式为Gemini格式
        
        Returns:
            (system_instruction, chat_history)
        """
        system_instruction = None
        chat_history = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                chat_history.append({
                    "role": "user",
                    "parts": [msg["content"]]
                })
            elif msg["role"] == "assistant":
                chat_history.append({
                    "role": "model",
                    "parts": [msg["content"]]
                })
        
        return system_instruction, chat_history
    
    def _convert_tools_to_gemini(
        self,
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List]:
        """转换OpenAI工具格式为Gemini Function Calling格式"""
        if not tools:
            return None
        
        gemini_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                gemini_tools.append({
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                })
        
        return gemini_tools if gemini_tools else None
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用Gemini生成文本"""
        try:
            logger.info(f"🔵 调用Gemini API - 模型: {model}")
            
            system_instruction, chat_history = self._convert_messages_to_gemini(messages)
            
            # 创建生成配置
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            # 创建模型实例
            model_kwargs = {
                "model_name": model,
                "generation_config": generation_config
            }
            
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            
            # 转换工具格式
            gemini_tools = self._convert_tools_to_gemini(tools)
            if gemini_tools:
                model_kwargs["tools"] = gemini_tools
            
            gemini_model = self.genai.GenerativeModel(**model_kwargs)
            
            # 如果有对话历史，使用chat模式
            if len(chat_history) > 1:
                # 最后一条消息是当前输入
                current_message = chat_history[-1]["parts"][0]
                history = chat_history[:-1]
                
                chat = gemini_model.start_chat(history=history)
                response = await chat.send_message_async(current_message)
            else:
                # 单条消息，直接生成
                current_message = chat_history[0]["parts"][0] if chat_history else ""
                response = await gemini_model.generate_content_async(current_message)
            
            # 处理响应
            content = ""
            tool_calls = []
            
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content += part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        # Gemini的function call
                        fc = part.function_call
                        tool_calls.append({
                            "id": f"call_{hash(fc.name)}",  # Gemini不提供call_id，生成一个
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": dict(fc.args)
                            }
                        })
            
            # 获取finish_reason
            finish_reason = "stop"
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            
            return {
                "content": content,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason
            }
            
        except Exception as e:
            logger.error(f"❌ Gemini API调用失败: {str(e)}")
            raise
    
    async def generate_text_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """使用Gemini流式生成文本"""
        try:
            logger.info(f"🔵 调用Gemini流式API - 模型: {model}")
            
            system_instruction, chat_history = self._convert_messages_to_gemini(messages)
            
            # 创建生成配置
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            # 创建模型实例
            model_kwargs = {
                "model_name": model,
                "generation_config": generation_config
            }
            
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            
            gemini_model = self.genai.GenerativeModel(**model_kwargs)
            
            # 如果有对话历史，使用chat模式
            if len(chat_history) > 1:
                current_message = chat_history[-1]["parts"][0]
                history = chat_history[:-1]
                
                chat = gemini_model.start_chat(history=history)
                response = await chat.send_message_async(
                    current_message,
                    stream=True
                )
            else:
                current_message = chat_history[0]["parts"][0] if chat_history else ""
                response = await gemini_model.generate_content_async(
                    current_message,
                    stream=True
                )
            
            # 流式输出
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
            
        except Exception as e:
            logger.error(f"❌ Gemini流式API调用失败: {str(e)}")
            raise


# ============================================================================
# Provider Factory
# ============================================================================

class AIProviderFactory:
    """AI提供商工厂"""
    
    @staticmethod
    def create_provider(
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ) -> AIProviderInterface:
        """创建AI提供商实例"""
        provider = provider.lower()
        
        if provider == AIProvider.OPENAI or provider == AIProvider.CUSTOM:
            return OpenAIProvider(api_key, base_url, http_client)
        elif provider == AIProvider.ANTHROPIC:
            return AnthropicProvider(api_key, base_url, http_client)
        elif provider == AIProvider.GEMINI:
            return GeminiProvider(api_key, base_url, http_client)
        else:
            raise ValueError(f"不支持的AI提供商: {provider}")


# ============================================================================
# Main AI Service
# ============================================================================

class AIService:
    """AI服务统一接口 - 支持多提供商"""
    
    def __init__(
        self,
        api_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None
    ):
        """初始化AI服务"""
        self.api_provider = api_provider or app_settings.default_ai_provider
        self.default_model = default_model or app_settings.default_model
        self.default_temperature = default_temperature or app_settings.default_temperature
        self.default_max_tokens = default_max_tokens or app_settings.default_max_tokens
        
        # 初始化提供商
        self.providers: Dict[str, AIProviderInterface] = {}
        
        # 初始化OpenAI
        openai_key = api_key if api_provider == "openai" else app_settings.openai_api_key
        if openai_key:
            try:
                base_url = api_base_url if api_provider == "openai" else app_settings.openai_base_url
                self.providers["openai"] = AIProviderFactory.create_provider(
                    "openai", openai_key, base_url
                )
            except Exception as e:
                logger.error(f"OpenAI提供商初始化失败: {e}")
        
        # 初始化Anthropic
        anthropic_key = api_key if api_provider == "anthropic" else app_settings.anthropic_api_key
        if anthropic_key:
            try:
                base_url = api_base_url if api_provider == "anthropic" else app_settings.anthropic_base_url
                self.providers["anthropic"] = AIProviderFactory.create_provider(
                    "anthropic", anthropic_key, base_url
                )
            except Exception as e:
                logger.error(f"Anthropic提供商初始化失败: {e}")
        
        # 初始化Gemini
        gemini_key = api_key if api_provider == "gemini" else app_settings.gemini_api_key
        if gemini_key:
            try:
                base_url = api_base_url if api_provider == "gemini" else app_settings.gemini_base_url
                self.providers["gemini"] = AIProviderFactory.create_provider(
                    "gemini", gemini_key, base_url
                )
            except Exception as e:
                logger.error(f"Gemini提供商初始化失败: {e}")
    
    def _get_provider(self, provider: Optional[str] = None) -> AIProviderInterface:
        """获取AI提供商实例"""
        provider = provider or self.api_provider
        provider = provider.lower()
        
        if provider not in self.providers:
            raise ValueError(f"提供商 '{provider}' 未初始化或不可用")
        
        return self.providers[provider]
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    async def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成文本（支持工具调用）"""
        provider_instance = self._get_provider(provider)
        model = model or self.default_model
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        messages = self._build_messages(prompt, system_prompt)
        
        return await provider_instance.generate_text(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice
        )
    
    async def generate_text_stream(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        provider_instance = self._get_provider(provider)
        model = model or self.default_model
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        
        messages = self._build_messages(prompt, system_prompt)
        
        async for chunk in provider_instance.generate_text_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            yield chunk
    
    async def generate_text_with_mcp(
        self,
        prompt: str,
        user_id: str,
        db_session,
        enable_mcp: bool = True,
        max_tool_rounds: int = 3,
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """支持MCP工具的AI文本生成（非流式）"""
        from app.services.mcp_tool_service import mcp_tool_service, MCPToolServiceError
        
        result = {
            "content": "",
            "tool_calls_made": 0,
            "tools_used": [],
            "finish_reason": "",
            "mcp_enhanced": False
        }
        
        # 获取MCP工具
        tools = None
        if enable_mcp:
            try:
                tools = await mcp_tool_service.get_user_enabled_tools(
                    user_id=user_id,
                    db_session=db_session
                )
                if tools:
                    logger.info(f"MCP增强: 加载了 {len(tools)} 个工具")
                    result["mcp_enhanced"] = True
            except MCPToolServiceError as e:
                logger.error(f"获取MCP工具失败: {e}")
                tools = None
        
        # 工具调用循环
        conversation_history = [{"role": "user", "content": prompt}]
        
        for round_num in range(max_tool_rounds):
            logger.info(f"MCP工具调用轮次: {round_num + 1}/{max_tool_rounds}")
            
            ai_response = await self.generate_text(
                prompt=conversation_history[-1]["content"],
                tools=tools if round_num == 0 else None,
                tool_choice=tool_choice if round_num == 0 else None,
                **kwargs
            )
            
            tool_calls = ai_response.get("tool_calls")
            
            if not tool_calls:
                result["content"] = ai_response.get("content", "")
                result["finish_reason"] = ai_response.get("finish_reason", "stop")
                break
            
            # 执行工具调用
            logger.info(f"AI请求调用 {len(tool_calls)} 个工具")
            
            try:
                tool_results = await mcp_tool_service.execute_tool_calls(
                    user_id=user_id,
                    tool_calls=tool_calls,
                    db_session=db_session
                )
                
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    if tool_name not in result["tools_used"]:
                        result["tools_used"].append(tool_name)
                
                result["tool_calls_made"] += len(tool_calls)
                
                tool_context = await mcp_tool_service.build_tool_context(
                    tool_results,
                    format="markdown"
                )
                
                next_prompt = f"{prompt}\n\n{tool_context}\n\n请基于以上工具查询结果，继续完成任务。"
                conversation_history.append({"role": "user", "content": next_prompt})
                
            except Exception as e:
                logger.error(f"执行MCP工具失败: {e}", exc_info=True)
                result["content"] = ai_response.get("content", "")
                result["finish_reason"] = "tool_error"
                break
        else:
            logger.warning(f"达到MCP最大调用轮次 {max_tool_rounds}")
            result["content"] = conversation_history[-1].get("content", "")
            result["finish_reason"] = "max_rounds"
        
        return result
    
    async def generate_text_stream_with_mcp(
        self,
        prompt: str,
        user_id: str,
        db_session,
        enable_mcp: bool = True,
        mcp_planning_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """支持MCP工具的AI流式文本生成（两阶段模式）"""
        from app.services.mcp_tool_service import mcp_tool_service
        
        enhanced_prompt = prompt
        
        if enable_mcp:
            try:
                tools = await mcp_tool_service.get_user_enabled_tools(
                    user_id=user_id,
                    db_session=db_session
                )
                
                if tools:
                    logger.info(f"MCP增强（流式）: 加载了 {len(tools)} 个工具")
                    
                    if not mcp_planning_prompt:
                        mcp_planning_prompt = (
                            f"任务: {prompt}\n\n"
                            f"请分析这个任务，决定是否需要查询外部信息。"
                            f"如果需要，请调用相应的工具获取信息。"
                        )
                    
                    planning_result = await self.generate_text_with_mcp(
                        prompt=mcp_planning_prompt,
                        user_id=user_id,
                        db_session=db_session,
                        enable_mcp=True,
                        max_tool_rounds=2,
                        tool_choice="auto",
                        **kwargs
                    )
                    
                    if planning_result["tool_calls_made"] > 0:
                        enhanced_prompt = (
                            f"{prompt}\n\n"
                            f"【参考资料】\n"
                            f"{planning_result.get('content', '')}"
                        )
                        logger.info(f"MCP工具规划完成，调用了 {planning_result['tool_calls_made']} 次工具")
            
            except Exception as e:
                logger.error(f"MCP工具规划失败: {e}")
        
        async for chunk in self.generate_text_stream(
            prompt=enhanced_prompt,
            **kwargs
        ):
            yield chunk


# ============================================================================
# Global Instances
# ============================================================================

ai_service = AIService()


def create_user_ai_service(
    api_provider: str,
    api_key: str,
    api_base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int
) -> AIService:
    """根据用户设置创建AI服务实例"""
    return AIService(
        api_provider=api_provider,
        api_key=api_key,
        api_base_url=api_base_url,
        default_model=model_name,
        default_temperature=temperature,
        default_max_tokens=max_tokens
    )