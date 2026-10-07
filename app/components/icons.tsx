// Small inline icons (stroke-based, inherit currentColor).

function Icon(props: { children: React.ReactNode }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {props.children}
    </svg>
  );
}

export const CloseIcon = () => (
  <Icon>
    <path d="M18 6 6 18M6 6l12 12" />
  </Icon>
);

export const DeleteIcon = () => (
  <Icon>
    <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6" />
  </Icon>
);

export const DownloadIcon = () => (
  <Icon>
    <path d="M12 3v12M7 10l5 5 5-5M5 21h14" />
  </Icon>
);

export const UploadIcon = () => (
  <Icon>
    <path d="M12 21V9M7 14l5-5 5 5M5 3h14" />
  </Icon>
);

export const DownIcon = () => (
  <Icon>
    <path d="m6 9 6 6 6-6" />
  </Icon>
);

export const EyeIcon = () => (
  <Icon>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" />
    <circle cx="12" cy="12" r="3" />
  </Icon>
);

export const EyeOffIcon = () => (
  <Icon>
    <path d="M3 3l18 18M10.6 5.1A10.4 10.4 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4.2M6.6 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7c1.7 0 3.2-.5 4.5-1.1M9.9 9.9a3 3 0 0 0 4.2 4.2" />
  </Icon>
);

export const LoadingIcon = () => (
  <svg width="22" height="8" viewBox="0 0 30 8" fill="currentColor" aria-hidden="true">
    {[3, 15, 27].map((cx, i) => (
      <circle key={cx} cx={cx} cy="4" r="3">
        <animate
          attributeName="opacity"
          values="0.2;1;0.2"
          dur="1s"
          begin={`${i * 0.2}s`}
          repeatCount="indefinite"
        />
      </circle>
    ))}
  </svg>
);
