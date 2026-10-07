"use client";

import React, { HTMLProps, useState } from "react";
import styles from "./ui.module.scss";
import { DownIcon, EyeIcon, EyeOffIcon } from "./icons";

export function IconButton(props: {
  onClick?: () => void;
  icon?: JSX.Element;
  type?: "primary" | "danger";
  text?: string;
  bordered?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      className={[
        styles["button"],
        props.bordered ? styles["bordered"] : "",
        props.type ? styles[props.type] : "",
      ].join(" ")}
      onClick={props.onClick}
      disabled={props.disabled}
      title={props.title}
    >
      {props.icon && <span className={styles["button-icon"]}>{props.icon}</span>}
      {props.text && <span>{props.text}</span>}
    </button>
  );
}

export function List(props: { children: React.ReactNode }) {
  return <div className={styles["list"]}>{props.children}</div>;
}

export function ListItem(props: {
  title: string;
  subTitle?: string;
  children?: React.ReactNode;
  vertical?: boolean;
}) {
  return (
    <div
      className={`${styles["list-item"]} ${props.vertical ? styles["vertical"] : ""}`}
    >
      <div className={styles["list-header"]}>
        <div>{props.title}</div>
        {props.subTitle && (
          <div className={styles["list-sub-title"]}>{props.subTitle}</div>
        )}
      </div>
      {props.children}
    </div>
  );
}

export function PasswordInput(props: HTMLProps<HTMLInputElement>) {
  const [visible, setVisible] = useState(false);
  return (
    <div className={styles["password"]}>
      <IconButton
        icon={visible ? <EyeIcon /> : <EyeOffIcon />}
        onClick={() => setVisible(!visible)}
      />
      <input {...props} type={visible ? "text" : "password"} />
    </div>
  );
}

export function Select(
  props: React.SelectHTMLAttributes<HTMLSelectElement>,
) {
  const { children, className, ...rest } = props;
  return (
    <div className={`${styles["select"]} ${className ?? ""}`}>
      <select {...rest}>{children}</select>
      <DownIcon />
    </div>
  );
}

export function showToast(message: string, delay = 3000) {
  const el = document.createElement("div");
  el.className = styles["toast"];
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), delay);
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className={styles["crash"]}>
          <h2>Something went wrong</h2>
          <pre>{this.state.error.message}</pre>
          <IconButton
            text="Reload"
            bordered
            onClick={() => location.reload()}
          />
        </div>
      );
    }
    return this.props.children;
  }
}
