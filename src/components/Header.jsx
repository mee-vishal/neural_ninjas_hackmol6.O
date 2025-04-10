import React from "react";
import { Link, NavLink } from "react-router-dom";

export default function Header() {
  return (
    <div className="navbar">
      <Link to="/" className="link">
        Home
      </Link>
      <Link to="/about" className="link">
        About Us
      </Link>
      <Link to="/help" className="link">
        Help
      </Link>
    </div>
  );
}
