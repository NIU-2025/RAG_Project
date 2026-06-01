import json
import base64
import hashlib
import hmac
import time
import asyncio
import httpx
from typing import Optional, TYPE_CHECKING
from app.core.logger import logger

if TYPE_CHECKING:
    from app.models.db import VoiceConfig


class VoiceASRService:

    @staticmethod
    async def recognize(file_path: str, provider: str, api_key: str, api_secret: str,
                       extra_params: Optional[dict] = None) -> str:
        """
        调用指定 Provider 的语音识别 API，返回识别文本。
        支持: baidu (百度), aliyun (阿里云)
        """
        if provider == "baidu":
            return await VoiceASRService._baidu_asr(file_path, api_key, api_secret, extra_params or {})
        elif provider == "aliyun":
            return await VoiceASRService._aliyun_asr(file_path, api_key, api_secret, extra_params or {})
        else:
            raise ValueError(f"不支持的语音 Provider: {provider}")

    # -------------------------------------------------------------------------
    # 百度智能云 ASR
    # 依赖: pip install baidu-aip
    # 参考: https://cloud.baidu.com/doc/SPEECH/s/QLyda114u
    # 参数说明:
    #   api_key    -> API Key
    #   extra_params.app_id    -> App ID
    #   extra_params.secret_key -> Secret Key
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_audio(file_path: str) -> str:
        """
        将音频统一转为 16kHz 单声道 PCM 文件路径返回。
        若已是 16kHz WAV/pcm，直接返回原路径。
        优先使用 ffmpeg；不可用时给出明确提示。
        """
        import os, tempfile

        '''
        - os.path.splitext(file_path) ：分割文件名和扩展名，返回元组（如 ("audio", ".mp3") ）
        - [1] ：取扩展名部分 .mp3
        - .lower() ：转为小写（ .MP3 → .mp3 ）
        - .lstrip(".") ：去掉前面的点（ .mp3 → mp3 ）
        '''
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")

        # 已是 PCM 且采样率为 16000，跳过
        if ext == "pcm":
            return file_path

        # WAV 需检查采样率，若已是 16kHz 也跳过
        if ext == "wav":
            try:
                import wave
                with wave.open(file_path, "rb") as wf:
                    '''
                    - wf.getframerate() ：获取采样率（Hz）
                    - wf.getnchannels() ：获取声道数（1 单声道）
                    '''
                    if wf.getframerate() == 16000 and wf.getnchannels() == 1:
                        return file_path
            except Exception:
                pass

        # 尝试 ffmpeg 转换
        try:
            import subprocess as _subprocess, uuid as _uuid, os as _os, tempfile as _tempfile
            # 先验证 ffmpeg 是否可用
            '''
            - _subprocess.run() ：执行子进程，返回进程对象
            - check=True ：如果进程返回非 0 状态码，抛出 CalledProcessError 异常
            - capture_output=True ：捕获进程标准输出和错误输出
            - timeout=10 ：设置超时时间（秒）
            '''
            _subprocess.run(
                ["ffmpeg", "-version"],
                check=True, capture_output=True, timeout=10,
            )
            tmp_dir = _tempfile.gettempdir() # 获取临时目录
            out_pcm = _os.path.join(tmp_dir, f"{_uuid.uuid4().hex}.pcm") # 生成唯一文件名
            
            '''
            - _subprocess.run() ：执行子进程，返回进程对象
            - capture_output=True ：捕获进程标准输出和错误输出
            - timeout=60 ：设置超时时间（秒）
            '''
            result = _subprocess.run(
                ["ffmpeg", "-y", "-i", file_path,
                 "-ac", "1", "-ar", "16000",
                 "-acodec", "pcm_s16le", "-f", "s16le",
                 out_pcm],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg 转换失败: {result.stderr.decode(errors='replace')}")
            logger.info(f"[voice] ffmpeg 转换成功: {file_path} -> {out_pcm}")
            return out_pcm
        except _subprocess.CalledProcessError as e:
            logger.error(f"[voice] ffmpeg 执行失败: {e.stderr.decode(errors='replace') if e.stderr else e}")
            raise RuntimeError(
                f"音频格式 {ext} 需要 ffmpeg 转换，但 ffmpeg 执行失败: {e}"
                "\n请确认 ffmpeg 已加入系统 PATH 并可在命令行执行 ffmpeg -version"
            ) from e
        except FileNotFoundError:
            logger.error("[voice] ffmpeg 未找到，请确认已安装并加入 PATH")
            raise RuntimeError(
                f"音频格式 {ext} 需要 ffmpeg 转换，但系统未找到 ffmpeg 命令。"
                "请安装 ffmpeg: https://ffmpeg.org/download.html"
                "\nWindows 可用 winget install ffmpeg 或下载预编译包并加入 PATH"
            )
        except Exception as e:
            logger.error(f"[voice] ffmpeg 不可用: {e}")
            raise RuntimeError(
                f"音频格式 {ext} 需要 ffmpeg 转换，但 ffmpeg 处理失败: {e}"
                "\n请确认 ffmpeg 已正确安装: https://ffmpeg.org/download.html"
                "\nWindows 可用 winget install ffmpeg 或下载预编译包并加入 PATH"
            ) from e

    @staticmethod
    async def _baidu_asr(file_path: str, api_key: str, api_secret: str,
                         params: dict) -> str:
        try:
            from aip import AipSpeech
        except ImportError:
            logger.error("[voice] 百度 SDK 未安装，请执行: pip install baidu-aip")
            raise RuntimeError("请安装百度 AIP SDK: pip install baidu-aip")

        app_id = params.get("app_id", "")
        if not app_id:
            logger.error("[voice] 百度 ASR 参数不完整: app_id 为空")
            raise RuntimeError("百度 ASR 配置不完整，请填写 App ID、API Key、Secret Key")

        dev_pid = params.get("dev_pid", 1537)

        # 统一音频格式：转为 16kHz 单声道 PCM，消除格式歧义
        normalized_path = VoiceASRService._normalize_audio(file_path)
        rate = 16000
        fmt = "pcm"
        logger.info(f"[voice] 音频转换: {file_path} -> {normalized_path}, format={fmt}, rate={rate}")

        try:
            client = AipSpeech(appId=app_id, apiKey=api_key, secretKey=api_secret)
            with open(normalized_path, "rb") as f:
                audio_data = f.read()

            def _sync_call():
                return client.asr(audio_data, fmt, rate, {"dev_pid": dev_pid})

            result = await asyncio.to_thread(_sync_call)

            if result.get("err_no"):
                msg = f"百度ASR失败 [{result.get('err_no')}]: {result.get('err_msg', '未知错误')}"
                logger.error(f"[voice] {msg}")
                raise RuntimeError(msg)

            result_list = result.get("result", [])
            if not result_list:
                logger.warning("[voice] 百度ASR未识别到文字内容（音频可能为空或无声）")
                raise ValueError("百度ASR未识别到文字内容")
            return "".join(result_list)
        finally:
            # 清理临时转换文件
            import os
            if normalized_path != file_path and os.path.exists(normalized_path):
                try:
                    os.remove(normalized_path)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # 阿里云语音识别（一句话识别 REST API）
    # 参考: https://help.aliyun.com/zh/isi/developer-reference/the-one-sentence-recognition-api
    # 参数说明:
    #   api_key    -> AccessKey ID
    #   api_secret -> AccessKey Secret
    #   extra_params.app_key -> AppKey
    # -------------------------------------------------------------------------
    @staticmethod
    async def _aliyun_asr(file_path: str, access_key_id: str, access_key_secret: str,
                          params: dict) -> str:
        app_key = params.get("app_key", "")
        if not app_key:
            logger.error("[voice] 阿里云 ASR 参数不完整: app_key 为空")
            raise RuntimeError("阿里云 ASR 配置不完整，请填写 AccessKey ID、AccessKey Secret、AppKey")

        # 统一音频格式：转为 16kHz 单声道 PCM
        normalized_path = VoiceASRService._normalize_audio(file_path)
        sample_rate = 16000
        fmt = "pcm"
        logger.info(f"[voice] 阿里云音频转换: {file_path} -> {normalized_path}, format={fmt}, rate={sample_rate}")

        try:
            with open(normalized_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("ascii")

            # 生成 Token（简化签名方式，适用于 NLS 1.0）
            token = VoiceASRService._generate_aliyun_token(access_key_id, access_key_secret)
            if not token:
                logger.error("[voice] 阿里云 Token 生成失败，请检查 AccessKey ID 和 AccessKey Secret 是否正确")
                raise RuntimeError("阿里云 Token 生成失败，请检查 AccessKey 配置")

            url = f"https://nls-gateway-cn-shanghai.aliyuncs.com/stream/语音+/vocoder_16k/pcm"
            payload = {
                "appkey": app_key,
                "format": "pcm",
                "sample_rate": sample_rate,
                "audio_base64": audio_base64,
            }

            headers = {
                "Content-Type": "application/json",
                "X-NLS-Token": token,
            }

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code == 401:
                raise RuntimeError("阿里云 Token 无效或已过期，请重新生成")
            resp.raise_for_status()
            data = resp.json()

            if data.get("error_code"):
                msg = f"阿里云ASR失败: {data.get('error_message', data)}"
                logger.error(f"[voice] {msg}")
                raise RuntimeError(msg)

            result_list = data.get("result", [])
            if not result_list:
                logger.warning("[voice] 阿里云ASR未识别到文字内容（音频可能为空或无声）")
                raise ValueError("阿里云ASR未识别到文字内容")
            return "".join(result_list)
        finally:
            # 清理临时转换文件
            import os
            if normalized_path != file_path and os.path.exists(normalized_path):
                try:
                    os.remove(normalized_path)
                except Exception:
                    pass

    @staticmethod
    def _generate_aliyun_token(access_key_id: str, access_key_secret: str) -> str:
        """
        生成阿里云 NLS API Token。
        使用签名方式获取访问 Token（参考阿里云文档）。
        """
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdkcore.profile import region_provider
            from aliyunsdknls.request.v20171225 import CreateTokenRequest
        except ImportError:
            logger.warning("aliyun-python-sdk-core-nls 未安装，尝试直接生成签名 Token")
            return VoiceASRService._aliyun_token_direct(access_key_id, access_key_secret)

        try:
            client = AcsClient(access_key_id, access_key_secret, "cn-shanghai")
            request = CreateTokenRequest.CreateTokenRequest()
            response = client.do_action_with_exception(request)
            resp = json.loads(response)
            return resp.get("Token", {}).get("Id", "")
        except Exception as e:
            logger.error(f"阿里云 Token 获取失败: {e}")
            return ""

    @staticmethod
    def _aliyun_token_direct(access_key_id: str, access_key_secret: str) -> str:
        """
        直接构造签名方式获取 Token（无 SDK 备用方案）。
        """
        timestamp = int(time.time() * 1000)
        nonce = hashlib.md5(str(timestamp).encode()).hexdigest()[:16]

        params = {
            "AccessKeyId": access_key_id,
            "Action": "CreateToken",
            "Format": "JSON",
            "SignatureMethod": "HMAC-SHA1",
            "SignatureNonce": nonce,
            "SignatureVersion": "1.0",
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp / 1000)),
            "Version": "2019-07-07",
        }
        sorted_params = sorted(params.items())
        canonical = "&".join(f"{k}={_percent_encode(str(v))}" for k, v in sorted_params)
        string_to_sign = f"GET&%2F&{_percent_encode(canonical)}"
        sig = base64.b64encode(
            hmac.new(
                (access_key_secret + "&").encode(),
                string_to_sign.encode(),
                hashlib.sha1
            ).digest()
        ).decode()
        params["Signature"] = sig

        query = "&".join(f"{k}={_percent_encode(str(v))}" for k, v in params.items())
        token_url = f"https://nls-gateway-cn-shanghai.aliyuncs.com/?{query}"

        try:
            import urllib.request
            with urllib.request.urlopen(token_url, timeout=10) as r:
                data = json.loads(r.read())
                return data.get("Token", {}).get("Id", "")
        except Exception:
            return ""

    @staticmethod
    async def recognize_from_config(file_path: str, cfg: "VoiceConfig") -> str:
        """根据 VoiceConfig 记录执行语音识别"""
        extra = json.loads(cfg.extra_params) if cfg.extra_params else {}
        text = await VoiceASRService.recognize(
            file_path=file_path,
            provider=cfg.provider,
            api_key=cfg.api_key or "",
            api_secret=cfg.api_secret or "",
            extra_params=extra,
        )
        logger.info(f"[voice_asr] 识别结果: {text}")
        return text


def _percent_encode(val: str) -> str:
    """URL Percent Encode (RFC 3986)"""
    import urllib.parse
    return urllib.parse.quote(str(val), safe="")
