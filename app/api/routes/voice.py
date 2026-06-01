from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json
from app.db.session import get_db
from app.models.db import User, VoiceConfig
from app.models.schemas import VoiceConfigOut, VoiceConfigUpdate, Resp
from app.api.deps import get_current_user, require_permission

router = APIRouter(prefix="/voice", tags=["语音配置"])

PROVIDERS = ["baidu", "aliyun"]


def _mask(val: str) -> str:
    if not val or len(val) <= 8:
        return "****"
    return val[:4] + "****" + val[-4:]


def _is_configured(provider: str, cfg: VoiceConfig | None, extra: dict) -> bool:
    if cfg is None:
        return False
    base_ok = bool(cfg.api_key and cfg.api_secret)
    if provider == "baidu":
        return base_ok and bool(extra.get("app_id"))
    if provider == "aliyun":
        return base_ok and bool(extra.get("app_key"))
    return base_ok


def _build_out(provider: str, cfg: VoiceConfig | None) -> VoiceConfigOut:
    extra = json.loads(cfg.extra_params) if cfg and cfg.extra_params else {}
    if cfg is None:
        return VoiceConfigOut(
            provider=provider,
            is_enabled=True,
            is_default=(provider == "baidu"),
            is_configured=_is_configured(provider, None, extra),
        )
    return VoiceConfigOut(
        id=cfg.id,
        provider=cfg.provider,
        api_key=_mask(cfg.api_key) if cfg.api_key else None,
        api_secret=_mask(cfg.api_secret) if cfg.api_secret else None,
        extra_params=cfg.extra_params,
        is_enabled=cfg.is_enabled,
        is_default=cfg.is_default,
        is_configured=_is_configured(provider, cfg, extra),
    )


@router.get("", response_model=Resp)
async def list_voice_configs(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("voice_config")),
):
    """列出所有语音 Provider 配置"""
    result = await db.execute(select(VoiceConfig))
    db_configs = {c.provider: c for c in result.scalars().all()}
    data = [_build_out(p, db_configs.get(p)) for p in PROVIDERS]
    return Resp(data=data)


@router.put("/{provider}", response_model=Resp)
async def upsert_voice_config(
    provider: str,
    body: VoiceConfigUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("voice_config")),
):
    """新增或更新语音 Provider 配置"""
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"不支持的 Provider: {provider}")

    result = await db.execute(select(VoiceConfig).where(VoiceConfig.provider == provider))
    cfg = result.scalar_one_or_none()

    if cfg is None:
        cfg = VoiceConfig(provider=provider)
        db.add(cfg)

    if body.api_key is not None:
        cfg.api_key = body.api_key.strip() or None
    if body.api_secret is not None:
        cfg.api_secret = body.api_secret.strip() or None
    if body.extra_params is not None:
        cfg.extra_params = body.extra_params.strip() or None
    if body.is_enabled is not None:
        cfg.is_enabled = body.is_enabled
    if body.is_default is not None:
        if body.is_default:
            # 取消其他默认
            all_result = await db.execute(select(VoiceConfig))
            for c in all_result.scalars().all():
                c.is_default = False
        cfg.is_default = body.is_default

    await db.flush()
    await db.refresh(cfg)
    return Resp(data=_build_out(provider, cfg), message="配置已保存")


@router.delete("/{provider}", response_model=Resp)
async def delete_voice_config(
    provider: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("voice_config")),
):
    """删除语音 Provider 配置（恢复默认）"""
    result = await db.execute(select(VoiceConfig).where(VoiceConfig.provider == provider))
    cfg = result.scalar_one_or_none()
    if cfg:
        await db.delete(cfg)
    return Resp(message="配置已清除")
